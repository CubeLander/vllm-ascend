import ast, math, sys, typing, unittest
from pathlib import Path
import torch

# Execute the exact installed helpers on CPU, without initializing a worker/NPU.
source = Path(sys.argv.pop(1)).read_text()
tree = ast.parse(source)
names = {'_make_int8_block_view', '_canonicalize_split_cache'}
ns = dict(torch=torch, math=math, Sequence=typing.Sequence)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<installed-offload-helpers>', 'exec'), ns)
canonical = ns['_canonicalize_split_cache']

class PackedKV(unittest.TestCase):
    def test_interleaved_restore_and_sentinels(self):
        for dtype in (torch.int8, torch.float32):
            for factor in (2, 16):
                for reverse in (False, True):
                    with self.subTest(dtype=dtype, factor=factor, reverse=reverse):
                        n, payload, lead, tail = 4, 32, 7, 11
                        base = torch.arange(lead+n*factor*2*payload+tail, dtype=dtype)
                        physical = base[lead:-tail].view(n*factor, 2, payload)
                        parts = [physical[:, 0], physical[:, 1]]
                        if reverse: parts.reverse()
                        total = factor*2*payload*base.element_size()
                        refs = canonical(parts, n, total)
                        self.assertEqual(len(refs), 1)
                        view, size = refs[0]
                        self.assertEqual(size, total)
                        self.assertEqual(view.data_ptr(), physical.data_ptr())
                        self.assertEqual(tuple(view.shape), (n,total))
                        expected = physical.reshape(n,-1).view(torch.int8)
                        self.assertTrue(torch.equal(view, expected))
                        original = base.clone()
                        for page in (0, n-1):
                            host = view[page].clone()
                            physical[page*factor:(page+1)*factor].zero_()
                            view[page].copy_(host)
                            self.assertTrue(torch.equal(base, original))

    def test_separate_contiguous_unchanged(self):
        k, v = torch.arange(24).reshape(8,3), torch.arange(24,48).reshape(8,3)
        refs = canonical([k,v],2,4*6*k.element_size())
        self.assertEqual(len(refs),2)
        self.assertTrue(torch.equal(refs[0][0],k.reshape(2,-1).view(torch.int8)))
        self.assertTrue(torch.equal(refs[1][0],v.reshape(2,-1).view(torch.int8)))

    def test_single_full_view_unchanged(self):
        full=torch.arange(48,dtype=torch.int8).reshape(8,6)
        self.assertTrue(torch.equal(canonical([full],2,24)[0][0],full.reshape(2,24)))

    def test_no_coalescing_keeps_separate_refs(self):
        p=torch.zeros((2,2,3),dtype=torch.int8)
        self.assertEqual(len(canonical([p[:,0],p[:,1]],2,6)),2)

    def test_reject_gaps(self):
        p=torch.zeros((8,3,3),dtype=torch.int8)
        with self.assertRaisesRegex(ValueError,'Cannot coalesce'):
            canonical([p[:,0],p[:,2]],2,24)

    def test_reject_internal_transpose(self):
        p=torch.zeros((8,2,2,3),dtype=torch.int8)
        with self.assertRaisesRegex(ValueError,'non-contiguous'):
            canonical([p[:,0].transpose(1,2),p[:,1].transpose(1,2)],2,48)

    def test_reject_too_few_physical_blocks(self):
        p=torch.zeros((7,2,3),dtype=torch.int8)
        with self.assertRaisesRegex(ValueError,'too few'):
            canonical([p[:,0],p[:,1]],2,24)

    def test_reject_overlapping_parts(self):
        p=torch.zeros((8,2,3),dtype=torch.int8)
        with self.assertRaisesRegex(ValueError,'Cannot coalesce'):
            canonical([p[:,0],p[:,0]],2,24)

unittest.main()

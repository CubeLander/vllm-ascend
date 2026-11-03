# Qwen3.5 native CPU-offload experiment snapshot

This branch preserves the exact packed-KV adaptation used in the Qwen3.5-35B-A3B
native CPU-offload experiments. It is not a general release qualification.

- vLLM-Ascend base: `c09d3e20b9c45c80b34ae3781440d90904a44023` (0.30.0).
- Companion vLLM: `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (0.30.0).
- Changed production file:
  `vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py`.
- Original file SHA256: `7e095997d34216045b25bf9b7af4fed6ed8360f18fa6108f07ff77504ccc1ae5`.
- Patched file SHA256: `3c264fad94e4f4cff5da6290639c5c722a3a32f070ffb14efef69a34f9c6dd34`.

The repair recognizes non-overlapping K/V views that exactly tile an interleaved
physical block, then coalesces whole packed blocks. Gaps, overlaps, non-contiguous
payloads and insufficient block counts remain errors. It does not change cache
policy, scheduling, copy kernels or precision.

Run the preserved standalone CPU regression against the actual target file:

```sh
python experiments/qwen35-native-offload-030/test_packed_kv.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py
```

The test extracts only the two layout helpers with AST and exercises them with
real CPU Torch tensors; it does not initialize the plugin or an NPU. Eight tests
passed again when archiving this branch. This is not a new device/model accuracy
run. Existing bounded restoration and complete session observations retain their
original source/environment scope. No weights, credentials or raw run output are
included here. Upstream Apache-2.0 attribution remains intact.

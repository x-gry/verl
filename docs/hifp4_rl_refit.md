# HiFP4 rollout weight refit

The veRL adapter supports W4A8_HIFP4_MXFP8 and W4A4_HIFP4, including the
existing HIF4 aliases. Pair the veRL changes with the vllm-ascend scheme
changes: the concrete schemes expose `rl_quant_type` and `group_size`.

Configure Actor MXFP8 independently, and set
`actor_rollout_ref.rollout.quantization=ascend`. Use a matching quantized model
and its existing quant_model_description.json (parameter names mapped to type
strings). For the Qwen3 acceptance setup, use matching 64-element HiFP4 groups.

Reload ordering is restore once, quantize/load each bucket, then process once.
Linear restores transposed views; MoE reverses the nibble packing permutation.
This does not dequantize to BF16 or implement residual error compensation.

## Operator integration

The veRL adapter imports `cann_ops_nn.ops` to register the operator, then calls
`torch.ops.cann_ops_nn.dynamic_hi_f4_quant(weight, axis=-1)` for each 2D matrix.
It returns packed uint8 weights and FP32 group scales. No dst_type or block_size
argument is passed. The scheme must use 64-element groups. Expert-first 3D
weights are quantized one expert at a time and stacked in checkpoint layout.

A trailing scale dimension is removed only when the output has shape
(N, ceil(K/64), 1), preserving logical size-1 dimensions. Model loaders retain
responsibility for TP/EP sharding. Inference layout handling remains in the
backend process/restore methods. No fallback to E2M1 is performed.

## Validation

CPU layout tests cover both scheme families, Linear/MoE repeated reloads,
Parameter identity, idempotency, and exact nibble round trips. veRL tests mock
the quant operator to check aliases, classification and scale contracts.

Run from the respective repository root with PyTorch available:

- veRL: `python tests/utils/test_hifp4_quant_on_cpu.py`
- vllm-ascend: `python tests/ut/quantization/methods/test_hifp4_rl_reload_on_cpu.py`

The authoring environment has no PyTorch; these tests could not run there.
Python syntax and git diff whitespace checks passed. Simulator validation with
Qwen3-30B/GSM8K and real operator ABI confirmation remain required. Simulation
success does not establish numeric accuracy or hardware performance. Existing
W4A8 fake-output paths are unchanged. Dynamic EPLB reload is outside this
initial validation scope.

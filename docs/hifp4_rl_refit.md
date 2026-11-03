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

The isolated veRL `_quantize_hifp4_weight` call currently assumes
`npu_dynamic_mx_quant(weight, axis=-1, dst_type=torch_npu.hifloat4_x2,
block_size=scheme.group_size)`. This is an integration assumption, not a
verified ABI for the simulation image. Verify the target torch_npu accepts
these keywords and returns packed bytes and FP32 group scales. Update this
single adapter if the supplied operator uses a different signature or scale
representation. No fallback to E2M1 is performed.

Scale shapes before model loading are (..., ceil(K/group_size)); a single
operator trailing scale dimension is removed without squeezing logical size-1
dimensions. Loader code retains responsibility for TP/EP sharding.

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

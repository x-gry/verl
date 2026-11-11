"""Unit tests for the HiFP4 refit adapter, with the NPU operator mocked."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


def load_helpers():
    path = Path(__file__).resolve().parents[2] / "verl/utils/vllm/vllm_quant_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_get_hifp4_scheme", "_quantize_hifp4_weight", "is_vllm_ascend_mx_quant", "_classify_quant_weight"}
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in names
    ) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HIFP4_QUANT_ALIASES" for t in n.targets)
    )]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class TestHiFP4Quant(unittest.TestCase):
    def test_alias_detection(self):
        scope = load_helpers()
        class Config:
            pass
        package = SimpleNamespace(AscendModelSlimConfig=Config)
        with patch.dict(sys.modules, {"vllm_ascend.quantization": package}):
            for name in (*scope["HIFP4_QUANT_ALIASES"], "W8A8_MXFP8", "W4A8_MXFP", "FLOAT"):
                config = Config()
                config.quant_description = {"layer.weight": name}
                self.assertEqual(scope["is_vllm_ascend_mx_quant"](config), name != "FLOAT")

    def test_scheme_precedes_uint8_fallback(self):
        scope = load_helpers()
        scheme = SimpleNamespace(rl_quant_type="W4A4_HIFP4")
        module = SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme))
        scope.update(is_fp8_weight=lambda *a: False, get_module_from_param_name=lambda *a: module,
                     is_w4a8_weight=lambda m: True)
        self.assertEqual(scope["_classify_quant_weight"]("layer.weight", None, True), "hifp4")
        scheme.rl_quant_type = "W4A8_MXFP"
        self.assertEqual(scope["_classify_quant_weight"]("layer.weight", None, True), "mx_fp4")
        self.assertIsNone(scope["_classify_quant_weight"]("layer.weight_scale", None, True))

    def call_with_op(self, helper, weight, op, group_size=64):
        from types import ModuleType

        package = ModuleType("cann_ops_nn")
        package.__path__ = []
        registration = ModuleType("cann_ops_nn.ops")
        with patch.dict(sys.modules, {"cann_ops_nn": package, "cann_ops_nn.ops": registration}), patch.object(
            torch.ops, "cann_ops_nn", SimpleNamespace(dynamic_hi_f4_quant=op)
        ):
            return helper(weight, SimpleNamespace(group_size=group_size))

    def test_scale_singleton_dimensions_are_preserved(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        for shape in ((1, 64), (3, 128), (256, 1024)):
            for dtype in (torch.bfloat16, torch.float16):
                weight = torch.zeros(shape, dtype=dtype)
                packed = torch.zeros((shape[0], shape[1] // 2), dtype=torch.uint8)
                scale_shape = (shape[0], shape[1] // 64)
                for trailing in (False, True):
                    scale = torch.ones((*scale_shape, 1) if trailing else scale_shape)
                    op = Mock(return_value=(packed, scale))
                    result, actual_scale = self.call_with_op(helper, weight, op)
                    self.assertEqual(tuple(actual_scale.shape), scale_shape)
                    self.assertIs(result, packed)
                    op.assert_called_once_with(weight, axis=-1)

    def test_experts_quantized_independently(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        weight = torch.arange(2 * 3 * 128).reshape(2, 3, 128).to(torch.bfloat16)
        outputs = [(torch.full((3, 64), i, dtype=torch.uint8), torch.full((3, 2, 1), float(i)))
                   for i in (1, 2)]
        op = Mock(side_effect=outputs)
        packed, scale = self.call_with_op(helper, weight, op)
        self.assertEqual(op.call_count, 2)
        for i in range(2):
            self.assertTrue(torch.equal(op.call_args_list[i].args[0], weight[i]))
            self.assertEqual(op.call_args_list[i].kwargs, {"axis": -1})
            self.assertTrue(torch.equal(packed[i], outputs[i][0]))
            self.assertTrue(torch.equal(scale[i], outputs[i][1].squeeze(-1)))

    def test_invalid_output_rejected(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        for packed, scale in (
            (torch.zeros(2, 32, dtype=torch.uint8), torch.ones(2, 2)),
            (torch.zeros(2, 32, dtype=torch.uint8), torch.ones(2, 1, dtype=torch.uint8)),
            (torch.zeros(2, 32), torch.ones(2, 1)),
        ):
            with self.assertRaises(ValueError):
                self.call_with_op(helper, torch.zeros(2, 64), Mock(return_value=(packed, scale)))

    def test_incompatible_group_rejected_before_operator_call(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        op = Mock()
        with self.assertRaisesRegex(ValueError, "group_size=64"):
            self.call_with_op(helper, torch.zeros(2, 64), op, group_size=32)
        op.assert_not_called()

    def test_operator_error_is_not_silently_replaced(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        with self.assertRaisesRegex(RuntimeError, "operator failure"):
            self.call_with_op(helper, torch.zeros(2, 64), Mock(side_effect=RuntimeError("operator failure")))


if __name__ == "__main__":
    unittest.main()

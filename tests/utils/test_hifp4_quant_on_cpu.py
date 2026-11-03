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

    def test_scale_singleton_dimensions_are_preserved(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        for shape in ((1, 64), (3, 128), (2, 1, 64)):
            with self.subTest(shape=shape):
                weight = torch.zeros(shape, dtype=torch.bfloat16)
                packed = torch.zeros((*shape[:-1], shape[-1] // 2), dtype=torch.uint8)
                scale_shape = (*shape[:-1], shape[-1] // 64)
                for trailing in (False, True):
                    scale = torch.ones((*scale_shape, 1) if trailing else scale_shape)
                    op = Mock(return_value=(packed, scale))
                    npu = SimpleNamespace(hifloat4_x2=object(), npu_dynamic_mx_quant=op)
                    with patch.dict(sys.modules, {"torch_npu": npu}):
                        result, actual_scale = helper(weight, SimpleNamespace(group_size=64))
                    self.assertEqual(tuple(actual_scale.shape), scale_shape)
                    self.assertEqual(result.dtype, torch.uint8)
                    op.assert_called_once_with(weight, axis=-1, dst_type=npu.hifloat4_x2, block_size=64)

    def test_invalid_scale_rejected(self):
        helper = load_helpers()["_quantize_hifp4_weight"]
        for scale in (torch.ones(2, 2), torch.ones(2, 1, dtype=torch.uint8)):
            npu = SimpleNamespace(hifloat4_x2=object(), npu_dynamic_mx_quant=Mock(
                return_value=(torch.zeros(2, 32, dtype=torch.uint8), scale)))
            with patch.dict(sys.modules, {"torch_npu": npu}), self.assertRaises(ValueError):
                helper(torch.zeros(2, 64), SimpleNamespace(group_size=64))


if __name__ == "__main__":
    unittest.main()

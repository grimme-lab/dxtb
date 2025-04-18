# This file is part of dxtb.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test the differentiable parameter model.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from tad_mctc.convert import str_to_device
from torch import nn

from dxtb import GFN1_XTB, GFN2_XTB, Param, ParamModule
from dxtb._src.constants.defaults import DEFAULT_BASIS_INT
from dxtb._src.param.module import NonNumericValue, ParameterModule
from dxtb._src.param.module.param import _convert, _revert


def test_forward() -> None:
    """Test the forward passes."""
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)

    tree = diff_param.forward()
    assert isinstance(tree, nn.ModuleDict)

    ##########################################

    x = torch.tensor([1.0, 2.0, 3.0])
    pm = ParameterModule(value=x)

    p = pm.forward()
    assert isinstance(p, nn.Parameter)
    assert p.shape == x.shape

    ##########################################

    val = NonNumericValue("test")
    assert isinstance(val, nn.Module)

    value = val.forward()
    assert value == "test"


def test_module_creation() -> None:
    """Test that a ParamModule can be created from a valid parameter model."""
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)

    tree = diff_param.forward()
    assert isinstance(tree, nn.ModuleDict)


def test_get_valid() -> None:
    """Test that a valid key path returns the underlying tensor."""
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)
    gexp = diff_param.get("charge", "effective", "gexp")

    msg = "gexp should be returned as a tensor."
    assert isinstance(gexp, torch.Tensor), msg

    # By default, parameters are created with gradients disabled.
    msg = "gexp should have requires_grad == False by default."
    assert gexp.requires_grad is False, msg


def test_get_invalid_key() -> None:
    """Test that using a non-existent key produces a KeyError."""
    par = GFN2_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)

    with pytest.raises(KeyError):
        diff_param.get("nonexistent")


def test_get_wrong_type() -> None:
    """
    Non-string key for ModuleDict (or non-int for ModuleList) raises TypeError.
    """
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)

    with pytest.raises(TypeError):
        diff_param.get(123)


def test_get_no_keys_provided() -> None:
    """Calling get without keys should raise KeyError."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    with pytest.raises(KeyError):
        diff.get()


def test_get_split_single_key() -> None:
    """Single string with dot splits into multiple keys."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    v1 = diff.get("element.C", unwrapped=False)
    v2 = diff.get("element", "C", unwrapped=False)
    assert v1 is v2


def test_get_modulelist_and_wrong_index_type() -> None:
    """Accessing ModuleList with correct and incorrect key types."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    shells = diff.get("element", "C", "shells", unwrapped=False)
    assert isinstance(shells, nn.ModuleList)

    # correct integer index
    item = diff.get("element", "C", "shells", 0)
    assert isinstance(item, str)
    assert item == "2s"

    # invalid string index for ModuleList
    with pytest.raises(TypeError):
        diff.get("element", "C", "shells", "bad")


def test_get_unwrapped_false_returns_module() -> None:
    """When unwrapped=False, ParameterModule is returned, not tensor."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    pm = diff.get("charge", "effective", "gexp", unwrapped=False)
    assert isinstance(pm, ParameterModule)


# differentiable


def test_set_differentiable_leaf() -> None:
    """
    Test that setting a leaf node (e.g. charge->effective->gexp) to be
    differentiable updates the tensor.
    """
    par = GFN2_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)

    # Retrieve gexp and verify default gradient flag.
    gexp = diff_param.get("charge", "effective", "gexp")

    msg = "gexp should initially be non-differentiable."
    assert gexp.requires_grad is False, msg

    # Set the leaf to be differentiable.
    diff_param.set_differentiable("charge", "effective", "gexp")
    gexp_after = diff_param.get("charge", "effective", "gexp")

    msg = "gexp should be set to differentiable."
    assert gexp_after.requires_grad is True, msg


def test_set_differentiable_branch_ignore_non_numeric() -> None:
    """
    Test that setting an entire branch (e.g. 'charge') to be differentiable
    with ``ignore_non_numeric=True`` recursively updates all numeric leaves,
    ignoring static leaves.
    """
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)

    # Set entire 'charge' branch to differentiable while ignoring static fields.
    diff_param.set_differentiable("charge", ignore_non_numeric=True)

    def check_diff(module: nn.Module) -> None:
        if isinstance(module, ParameterModule):
            msg = "All numeric parameters should be differentiable."
            assert module.param.requires_grad is True, msg
        elif isinstance(module, (nn.ModuleDict, nn.ModuleList)):
            for child in module.children():
                check_diff(child)
        # StaticValue is ignored.

    branch = diff_param.get("charge", unwrapped=False)
    check_diff(branch)


def test_set_differentiable_failure() -> None:
    """
    Test that attempting to set a non-numeric (static) field to be
    differentiable raises a TypeError when ignore_non_numeric is False.
    For example, 'meta'->'name' is expected to be a static string.
    """
    par = GFN2_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.double)
    with pytest.raises(TypeError):
        diff_param.set_differentiable("meta", "name", ignore_non_numeric=False)


def test_set_differentiable_key_errors() -> None:
    """
    Setting differentiable on invalid paths should raise KeyError or TypeError.
    """
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    # wrong key type
    with pytest.raises(TypeError):
        diff.set_differentiable(1)
    # non-existent path
    with pytest.raises(KeyError):
        diff.set_differentiable("no", "path")


def test_set_differentiable_non_numeric_failure() -> None:
    """
    Attempt to set a NonNumericValue leaf with ignore_non_numeric=False fails.
    """
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    with pytest.raises(TypeError):
        diff.set_differentiable("meta", "name", ignore_non_numeric=False)


def test_set_differentiable_on_modulelist_with_no_ignore() -> None:
    """
    Setting differentiability on ModuleList branch without ignoring
    non-numeric raises TypeError.
    """
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    # 'shells' is ModuleList of NonNumericValue
    with pytest.raises(TypeError):
        diff.set_differentiable(
            "element", "C", "shells", ignore_non_numeric=False
        )


def test_get_nonnumeric_leaf() -> None:
    """Accessing a non-numeric leaf returns its raw value."""
    par = GFN1_XTB.model_copy(deep=True)
    diff = ParamModule(par)

    meta = par.meta
    assert meta is not None
    expected = meta.name

    val = diff.get("meta", "name")
    assert isinstance(val, str)
    assert val == expected


def test_set_differentiable_branch_ignore_non_numeric_all_numeric() -> None:
    """
    Setting differentiable on a branch ignores non-numeric and updates all
    numeric leaves.
    """
    diff = ParamModule(GFN1_XTB.model_copy(deep=True), dtype=torch.double)
    diff.set_differentiable("charge", ignore_non_numeric=True)

    # Recursively check that all ParameterModule leaves require gradients
    def _check(mod):
        if isinstance(mod, ParameterModule):
            assert mod.param.requires_grad is True
        elif isinstance(mod, (nn.ModuleDict, nn.ModuleList)):
            for c in mod.children():
                _check(c)

    branch = diff.get("charge", unwrapped=False)
    _check(branch)


# convert and revert


def test_convert_and_revert_edge_cases() -> None:
    """
    Test _convert and _revert with various types and nested structures using
    approximate comparison for floats.
    """
    data = {
        "num": 5,
        "flt": 3.14,
        "lst_num": [1, 2.0, 3],
        "lst_mixed": [1, "a", 2.5],
        "nested": {"x": 1, "y": ["b", 2]},
    }
    mod = _convert(data, device=torch.device("cpu"), dtype=torch.double)
    out = _revert(mod)

    # Exact matches for integers and strings
    assert out["num"] == data["num"]
    assert out["lst_num"][0] == data["lst_num"][0]
    assert out["lst_num"][2] == data["lst_num"][2]
    assert out["lst_mixed"][0] == data["lst_mixed"][0]
    assert out["lst_mixed"][1] == data["lst_mixed"][1]
    assert out["nested"]["x"] == data["nested"]["x"]
    assert out["nested"]["y"][0] == data["nested"]["y"][0]
    assert out["nested"]["y"][1] == data["nested"]["y"][1]

    # Approximate matches for floats
    assert out["flt"] == pytest.approx(data["flt"])
    assert out["lst_num"][1] == pytest.approx(data["lst_num"][1])
    assert out["lst_mixed"][2] == pytest.approx(data["lst_mixed"][2])


def test_to_dict_and_pydantic_roundtrip() -> None:
    """Round-trip parameters through dict and Pydantic."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    d = diff.to_dict()
    assert isinstance(d, dict)

    p = diff.to_pydantic()
    assert isinstance(p, Param)

    d1 = GFN1_XTB.model_dump()
    d2 = p.model_dump()

    # Deep compare dictionaries, using approximate checks for float values
    def _compare(a, b):
        assert type(a) is type(b)
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys()
            for k in a:
                _compare(a[k], b[k])
        elif isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                _compare(x, y)
        elif isinstance(a, float) and isinstance(b, float):
            assert b == pytest.approx(a)
        else:
            assert a == b

    _compare(d1, d2)


def test_write_to_file() -> None:
    """Test writing to file in different formats."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "out.json"
        diff.to_file(json_path)
        assert json_path.exists()

        json_path = Path(tmpdir) / "out2.json"
        diff.to_json_file(json_path)
        assert json_path.exists()

        yaml_path = Path(tmpdir) / "out.yaml"
        diff.to_yaml_file(yaml_path)
        assert yaml_path.exists()

        toml_path = Path(tmpdir) / "out.toml"
        diff.to_toml_file(toml_path)
        assert toml_path.exists()


def test_contains_is_none_is_false() -> None:
    """Check __contains__, is_none, and is_false behaviors deterministically."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    assert "charge" in diff
    assert diff.is_none("no_branch")
    assert diff.is_false("thirdorder.shell")


def test_get_elem_param_pair_val_angular_pqn_valence() -> None:
    """Exercise all element- and pair-based accessors."""
    diff = ParamModule(GFN1_XTB.model_copy(deep=True))
    unique = torch.tensor([1, 6], dtype=torch.int64)
    t1 = diff.get_elem_param(unique, "gam3")
    assert isinstance(t1, torch.Tensor) and t1.shape == (2,)

    mat = diff.get_pair_param([1, 6])
    assert mat.shape == (2, 2)
    assert torch.allclose(mat, mat.T)

    ang = diff.get_elem_angular()
    assert isinstance(ang, dict)
    assert set(ang.keys()).issuperset({1, 6})

    val_mask = diff.get_elem_valence(unique)
    assert val_mask.dtype == torch.bool

    pqn = diff.get_elem_pqn(unique)
    assert pqn.dtype == DEFAULT_BASIS_INT


def test_convert_list_of_strings_returns_modulelist() -> None:
    """Non-numeric list input returns a ModuleList of NonNumericValue."""
    data = ["a", "b", "c"]
    mod = _convert(data, device=torch.device("cpu"), dtype=torch.float)
    assert isinstance(mod, nn.ModuleList)
    assert len(mod) == len(data)
    for idx, item in enumerate(mod):
        assert isinstance(item, NonNumericValue)
        assert item.value == data[idx]


def test_revert_raw_tensor_and_wrapped_tensor() -> None:
    """Raw tensor and NonNumericValue-wrapped tensor revert correctly."""
    t = torch.tensor([[1, 2], [3, 4]])
    # Raw tensor should convert to nested lists
    out_raw = _revert(t)
    assert out_raw == [[1, 2], [3, 4]]

    # Wrapped tensor returns raw tensor (per implementation)
    wrapped = NonNumericValue(t)
    out_wrapped = _revert(wrapped)
    assert isinstance(out_wrapped, torch.Tensor)
    assert torch.equal(out_wrapped, t)

    # Further reverting the raw tensor yields nested lists
    out_wrapped_list = _revert(out_wrapped)
    assert out_wrapped_list == [[1, 2], [3, 4]]


def test_revert_plain_python_type() -> None:
    """Test that a plain Python type (e.g. int) reverts correctly."""
    data = 42
    reverted = _revert(data)
    assert reverted == data
    assert isinstance(reverted, int)

    # Test with a string
    data_str = "Hello"
    reverted_str = _revert(data_str)
    assert reverted_str == data_str
    assert isinstance(reverted_str, str)


# Device and Dtype


def test_device() -> None:
    """Test that the device of the parameters is set correctly."""
    par = GFN1_XTB.model_copy(deep=True)
    param = ParamModule(par, device=torch.device("cpu"))
    assert param.device == torch.device("cpu")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_dtype(dtype: torch.dtype) -> None:
    """Test that the dtype of the parameters is set correctly."""
    par = GFN1_XTB.model_copy(deep=True)
    param = ParamModule(par, dtype=dtype)
    assert param.dtype == dtype


def test_dd() -> None:
    """Test that the dd of the parameters is set correctly."""
    par = GFN1_XTB.model_copy(deep=True)
    param = ParamModule(par, device=torch.device("cpu"), dtype=torch.double)
    assert param.dd == {"device": torch.device("cpu"), "dtype": torch.double}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Test changing the dtype."""
    param = ParamModule(GFN1_XTB, dtype=dtype)

    param = param.type(dtype)
    assert param.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    """Test changing the device."""
    device = str_to_device(device_str)
    param = ParamModule(GFN1_XTB, device=device)

    param = param.to(device)
    assert param.device == device

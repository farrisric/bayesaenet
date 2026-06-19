"""Verify that Pyro param store is moved to the correct device on test/predict start.

Regression test for the bug where BNN_Forces.on_test_start skipped
param_store_to when bnn_net already existed (after trainer.fit),
leaving guide weights on CPU while input data was on CUDA.
"""

import inspect
import textwrap

import pyro
import pytest
import torch

from bnn_aenet.models.utils import param_store_to


def test_param_store_to_moves_tensors():
    """param_store_to should move every tensor to the requested device."""
    pyro.clear_param_store()
    pyro.param("dummy_loc", torch.randn(4))
    pyro.param("dummy_scale", torch.randn(4))

    param_store_to("cpu")

    ps = pyro.get_param_store()
    for name in ps:
        val = ps[name]
        assert val.device == torch.device("cpu"), f"param '{name}' not on cpu after param_store_to"
    pyro.clear_param_store()


def _method_calls_param_store_to_unconditionally(cls, method_name):
    """Return True if param_store_to appears at the top indentation level
    of *method_name* (i.e. not nested inside an if-block)."""
    src = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
    in_if_block = False
    for line in src.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""'):
            continue
        indent = len(line) - len(stripped)
        if stripped.startswith("if "):
            in_if_block = True
            if_indent = indent
            continue
        if in_if_block and indent > if_indent:
            if "param_store_to" in stripped:
                return False
            continue
        in_if_block = False
        if "param_store_to" in stripped:
            return True
    return False


class TestBNNForcesParamStoreDevice:
    """on_test_start and on_predict_start must always call param_store_to."""

    def test_on_test_start_calls_param_store_to_unconditionally(self):
        from bnn_aenet.models.bnn_forces import BNN_Forces

        assert _method_calls_param_store_to_unconditionally(
            BNN_Forces, "on_test_start"
        ), "param_store_to must be called outside the `if bnn_net` block"

    def test_on_predict_start_calls_param_store_to_unconditionally(self):
        from bnn_aenet.models.bnn_forces import BNN_Forces

        assert _method_calls_param_store_to_unconditionally(
            BNN_Forces, "on_predict_start"
        ), "param_store_to must be called outside the `if bnn_net` block"

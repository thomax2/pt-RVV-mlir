import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "export") or not hasattr(torch.library, "custom_op"):
    pytest.skip("requires modern torch.export and torch.library.custom_op",
                allow_module_level=True)

from pt_model.neural_gm_splitcov_v33 import build_default_model_v33
from python.eot_frontend_export import EotFrontendExportModule


def _inputs(nmax=64):
    torch.manual_seed(23)
    return (
        torch.randn(1, nmax, 2), torch.randn(1, 2), torch.randn(1, 12),
        torch.rand(1, nmax) > 0.25, torch.randn(1, 128),
    )


def test_wrapper_matches_original_frontend():
    model = build_default_model_v33().eval()
    wrapper = EotFrontendExportModule(model).eval()
    inputs = _inputs()
    with torch.no_grad():
        expected, hidden = model.measurement_step(
            inputs[0], inputs[1], inputs[2], inputs[3], {"c": inputs[4]})
        actual = wrapper(*inputs)
    for lhs, rhs in zip(actual, (expected["pi"], expected["c_mean"],
                                 expected["c_logvar"], hidden["c"])):
        torch.testing.assert_close(lhs, rhs, atol=2e-6, rtol=2e-5)


def test_strict_export_retains_all_eot_nodes():
    wrapper = EotFrontendExportModule(build_default_model_v33().eval()).eval()
    inputs = _inputs(16)
    exported = torch.export.export(wrapper, inputs, strict=True)
    graph = str(exported.graph_module.graph)
    for name in ("point_features", "masked_mean", "masked_variance",
                 "masked_max", "masked_min", "gru_cell", "gm_parameterize"):
        assert f"eot.{name}" in graph or f"eot::{name}" in graph
    with torch.no_grad():
        eager = wrapper(*inputs)
        compiled = exported.module()(*inputs)
    for lhs, rhs in zip(eager, compiled):
        torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("steps", [10, 100])
def test_multiframe_gru_does_not_drift(steps):
    model = build_default_model_v33().eval()
    wrapper = EotFrontendExportModule(model).eval()
    points, sensor, state, mask, hidden_ref = _inputs(8)
    hidden_eot = hidden_ref.clone()
    with torch.no_grad():
        for _ in range(steps):
            expected, hidden = model.measurement_step(
                points, sensor, state, mask, {"c": hidden_ref})
            actual = wrapper(points, sensor, state, mask, hidden_eot)
            hidden_ref = hidden["c"]
            hidden_eot = actual[3]
    torch.testing.assert_close(hidden_eot, hidden_ref, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual[0], expected["pi"], atol=3e-6, rtol=3e-5)

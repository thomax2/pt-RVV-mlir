import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch.library, "custom_op") or not hasattr(torch.library, "opcheck"):
    pytest.skip("requires a PyTorch release with custom_op and opcheck",
                allow_module_level=True)

from python import eot_export_ops as eot


def _inputs(batch=2, points=8, channels=5):
    torch.manual_seed(7)
    data = torch.randn(batch, points, channels)
    mask = torch.tensor([[True, False, True, True, False, True, False, True],
                         [False, False, False, False, False, False, False, False]])
    return data, mask


def test_point_features_reference_and_origin():
    points = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    mask = torch.tensor([[True, False]])
    result = eot.point_features(points, mask, 1e-8, 1e-10)
    assert result.shape == (1, 2, 9)
    torch.testing.assert_close(result[0, 0, 5], torch.tensor(0.0))
    torch.testing.assert_close(result[0, 1], torch.zeros(9))


def test_masked_reductions_and_empty_values():
    data, mask = _inputs()
    weight = mask.to(data.dtype).unsqueeze(-1)
    count = weight.sum(1).clamp_min(1)
    mean = (data * weight).sum(1) / count
    variance = ((data - mean.unsqueeze(1)).square() * weight).sum(1) / count
    torch.testing.assert_close(eot.masked_mean(data, mask), mean)
    torch.testing.assert_close(eot.masked_variance(data, mask), variance)
    assert torch.equal(eot.masked_mean(data, mask)[1], torch.zeros(5))
    assert torch.equal(eot.masked_variance(data, mask)[1], torch.zeros(5))
    assert torch.equal(eot.masked_max(data, mask, -1e9)[1], torch.full((5,), -1e9))
    assert torch.equal(eot.masked_min(data, mask, 1e9)[1], torch.full((5,), 1e9))


def test_gru_matches_nn_gru_cell():
    torch.manual_seed(11)
    cell = torch.nn.GRUCell(6, 4)
    x, h = torch.randn(3, 6), torch.randn(3, 4)
    expected = cell(x, h)
    actual = eot.gru_cell(x, h, cell.weight_ih, cell.weight_hh,
                          cell.bias_ih, cell.bias_hh)
    torch.testing.assert_close(actual, expected)


def test_gm_shapes_and_ranges():
    outputs = eot.gm_parameterize(
        torch.randn(2, 4), torch.randn(2, 1), torch.randn(2, 2),
        torch.randn(2, 2), torch.rand(2, 1), 0.003, 0.7, 1.0, 100.0,
        1.0, -6.0, 1.5)
    assert [tuple(value.shape) for value in outputs] == [
        (2, 4), (2, 1), (2, 4), (2, 1), (2, 1),
        (2, 2), (2, 2), (2, 2), (2, 2),
    ]
    torch.testing.assert_close(outputs[0].sum(1), torch.ones(2))
    assert torch.all((outputs[5] >= 0.003) & (outputs[5] <= 0.7))


def test_all_custom_ops_pass_opcheck():
    data, mask = _inputs()
    points = torch.randn(2, 8, 2)
    cell = torch.nn.GRUCell(5, 4)
    x, h = torch.randn(2, 5), torch.randn(2, 4)
    cases = (
        (eot.point_features, (points, mask, 1e-8, 1e-10)),
        (eot.masked_mean, (data, mask)),
        (eot.masked_variance, (data, mask)),
        (eot.masked_max, (data, mask, -1e9)),
        (eot.masked_min, (data, mask, 1e9)),
        (eot.gru_cell, (x, h, cell.weight_ih, cell.weight_hh,
                        cell.bias_ih, cell.bias_hh)),
        (eot.gm_parameterize,
         (torch.randn(2, 4), torch.randn(2, 1), torch.randn(2, 2),
          torch.randn(2, 2), torch.rand(2, 1), 0.003, 0.7, 1.0,
          100.0, 1.0, -6.0, 1.5)),
    )
    for operator, args in cases:
        torch.library.opcheck(operator, args)


def test_shape_errors_are_explicit():
    with pytest.raises((ValueError, RuntimeError), match="shape"):
        eot.point_features(torch.randn(2, 3), torch.ones(2, dtype=torch.bool),
                           1e-8, 1e-10)

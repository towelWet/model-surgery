"""Projection removes the output direction on rectangular weights; input axes are unequal."""
import mlx.core as mx
from pipeline.surgery.abliterate_parts.projection.apply import project_out


def test_projection_output_axis_and_invalid_directions():
    weight = mx.arange(15, dtype=mx.float32).reshape(3, 5)
    direction = mx.array([1.0, 2.0, -1.0])
    actual = project_out(weight, direction, 1)
    assert float(mx.max(mx.abs(direction @ actual)).item()) < 1e-5
    assert bool(mx.allclose(project_out(weight, direction, 0), weight).item())
    for bad in (mx.zeros(3), mx.ones(5), mx.array([float("nan"), 1, 2])):
        try:
            project_out(weight, bad, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid output-space direction accepted")

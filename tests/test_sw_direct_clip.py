import numpy as np
import pytest
import xarray as xr

pytest.importorskip('git')   # GitPython and cdo are imported by TopoPyScale.topo_scale
pytest.importorskip('cdo')

from TopoPyScale import topo_scale as ts
from synthetic_steep_point import make_case, expected_sw_direct, S0, SSRD_WM2


def run(tmp_path):
    args = make_case(tmp_path)
    ts.pt_downscale_radiations(*args)
    return args, xr.open_dataset(tmp_path / 'downscaled' / 'down_pt_0.nc').load()


def test_hourly_ssrd_above_swtoa_in_the_synthetic_case(tmp_path):
    # guard: the case must actually contain the situation that triggered the bug
    ds_solar = make_case(tmp_path)[1]
    daylight = ~ds_solar.sunset.values
    assert (SSRD_WM2[daylight] > ds_solar.SWtoa.values[daylight]).any()


def test_sw_direct_never_exceeds_solar_constant_on_steep_slope(tmp_path):
    _, out = run(tmp_path)
    assert np.isfinite(out.SW_direct.values).all()
    assert out.SW_direct.max() <= S0
    assert out.SW.max() <= S0 + 1.


def test_sw_direct_matches_reference(tmp_path):
    (row, ds_solar, *_), out = run(tmp_path)
    expected = expected_sw_direct(ds_solar, row)
    np.testing.assert_allclose(out.SW_direct.values, expected, rtol=1e-3, atol=1e-2)


def test_unaffected_hours_are_unchanged(tmp_path):
    # hours where ssrd < SWtoa must not be modified by the cap: SW_direct = (SW - SW_diffuse) * cos_i / mu0
    (row, ds_solar, *_), out = run(tmp_path)
    ok = (~ds_solar.sunset.values) & (SSRD_WM2 < ds_solar.SWtoa.values)
    assert ok.sum() >= 3
    np.testing.assert_allclose(out.SW_direct.values[ok], expected_sw_direct(ds_solar, row)[ok], rtol=1e-3)

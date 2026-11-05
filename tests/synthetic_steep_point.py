"""
Synthetic steep, west-facing point used to test the downscaling of shortwave radiation.

Seven hourly time steps of a summer evening at ~39 N: the sun goes from 30 to 89.5 degrees zenith in the west.
ERA5 ssrd is an accumulation over the previous hour, so in the last hours before sunset ssrd / 3600 is larger than the
instantaneous top-of-atmosphere flux SWtoa (see SSRD_WM2 against SWTOA below). Without capping SW_direct at SWtoa this
produces a negative ka and, on a slope facing the sun, a direct flux above the solar constant.
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from types import SimpleNamespace

S0 = 1370.  # solar constant used by TopoPyScale [W m-2]
TIME = pd.date_range('2023-06-21 21:00', periods=7, freq='1h')   # UTC, i.e. afternoon/evening at UTC-7
ZENITH_DEG = np.array([30., 55., 70., 80., 85., 88., 89.5])
AZIMUTH_DEG = np.array([-40., -70., -85., -95., -103., -110., -115.])  # Dozier convention: 0 south, west negative
SSRD_WM2 = np.array([900., 650., 400., 215., 160., 110., 0.])           # ssrd / 3600, hour mean horizontal global SW
SLOPE_DEG = 30.
ASPECT_RAD = 3 * np.pi / 2   # west, in the [0, 2pi) range used by TopoPyScale
SVF = 0.95
P_PA = 80000.


def make_case(workdir):
    """
    Write the tmp/ files read by `pt_downscale_radiations` and return its arguments.

    Args:
        workdir (Path): output directory, `tmp/` and `downscaled/` are created in it

    Returns:
        tuple: (row, ds_solar, horizon_da, meta, output_dir) to call `pt_downscale_radiations(*args)`
    """
    workdir = Path(workdir)
    (workdir / 'tmp').mkdir(parents=True, exist_ok=True)
    (workdir / 'downscaled').mkdir(parents=True, exist_ok=True)
    n = len(TIME)

    zen = np.deg2rad(ZENITH_DEG)
    mu0 = np.cos(zen) * (np.cos(zen) > 0)
    ds_solar = xr.Dataset(
        {'zenith': ('time', zen),
         'azimuth': ('time', np.deg2rad(AZIMUTH_DEG)),
         'elevation': ('time', np.pi / 2 - zen),
         'mu0': ('time', mu0),
         'SWtoa': ('time', S0 * mu0),
         'sunset': ('time', mu0 < np.cos(np.deg2rad(89)))},
        coords={'time': TIME})

    down_pt = xr.Dataset(
        {'t': ('time', np.full(n, 290.)), 'q': ('time', np.full(n, 0.005)), 'p': ('time', np.full(n, P_PA))},
        coords={'time': TIME, 'level': ('time', np.full(n, 800.))})
    surf_interp = xr.Dataset(
        {'t2m': ('time', np.full(n, 290.)), 'q': ('time', np.full(n, 0.005)), 'sp': ('time', np.full(n, P_PA)),
         'strd': ('time', np.full(n, 300. * 3600)), 'ssrd': ('time', SSRD_WM2 * 3600)},
        coords={'time': TIME})
    down_pt.to_netcdf(workdir / 'tmp' / 'down_pt_0.nc', engine='h5netcdf')
    surf_interp.to_netcdf(workdir / 'tmp' / 'surf_interp_0.nc', engine='h5netcdf')

    row = SimpleNamespace(point_name=0, x=0., y=0., slope=np.deg2rad(SLOPE_DEG), aspect=ASPECT_RAD, svf=SVF)
    azi = np.arange(-175., 180., 10.)
    horizon_da = xr.DataArray(np.zeros((azi.size, 1, 1)), dims=('azimuth', 'y', 'x'),
                              coords={'azimuth': azi, 'y': [0.], 'x': [0.]}, name='horizon')   # flat horizon
    meta = {'file_pattern': 'down_pt_*.nc', 'output_directory': workdir, 'lw_terrain_flag': False}
    return row, ds_solar, horizon_da, meta, workdir


def expected_sw_direct(ds_solar, row, ssrd_wm2=SSRD_WM2):
    """
    Independent reference for SW_direct (numpy). Because ka is built from the same SW_direct, the Beer's law
    step in the code collapses to tmp = SW_direct, and the output is min(SW_direct, SWtoa) * cos_illumination / mu0.
    """
    zen, az, mu0 = ds_solar.zenith.values, ds_solar.azimuth.values, ds_solar.mu0.values
    toa, sun = ds_solar.SWtoa.values, ~ds_solar.sunset.values
    with np.errstate(all='ignore'):
        kt = np.clip(np.where(sun, ssrd_wm2 / toa, 0.), 0, 1)
        kd = 0.952 - 1.041 * np.exp(-np.exp(2.3 - 4.702 * kt))
        direct = np.minimum(ssrd_wm2 * (1 - kd), toa)
        cos_i = np.clip(mu0 * np.cos(row.slope) + np.sin(zen) * np.sin(row.slope) * np.cos(az - row.aspect), 0, None)
        out = np.where(sun, direct * cos_i / mu0, 0.)
    return out

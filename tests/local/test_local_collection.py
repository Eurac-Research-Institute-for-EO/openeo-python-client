import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from openeo.local import LocalConnection
except ImportError:
    LocalConnection = None


@pytest.mark.skipif(
    not LocalConnection, reason="environment does not support localprocessing"
)
def test_local_collection_metadata(tmp_path_factory):
    sample_netcdf = create_local_data(tmp_path_factory,2,2,2,'netcdf')
    sample_geotiff = create_local_data(tmp_path_factory,2,2,2,'tiff')
    local_conn = LocalConnection(sample_netcdf.as_posix())
    assert len(local_conn.list_collections()) == 1
    local_conn = LocalConnection([sample_netcdf.as_posix(),sample_geotiff.as_posix()])
    assert len(local_conn.list_collections()) == 2


@pytest.mark.skipif(
    not LocalConnection, reason="environment does not support localprocessing"
)
def test_load_local_collection_returns_dataset(tmp_path_factory):
    """load_local_collection returns Dataset for NetCDF files."""
    from openeo.local.processing import load_local_collection
    sample_dir = create_local_data(tmp_path_factory, 2, 2, 2, 'netcdf')
    result = load_local_collection(id=str(sample_dir / 'sample_data.nc'))
    assert isinstance(result, xr.Dataset)
    assert 'temperature' in result.data_vars
    assert 'precipitation' in result.data_vars


@pytest.mark.skipif(
    not LocalConnection, reason="environment does not support localprocessing"
)
def test_load_local_collection_tiff_returns_dataset(tmp_path_factory):
    """load_local_collection returns Dataset for GeoTIFF files."""
    from openeo.local.processing import load_local_collection
    sample_dir = create_local_data(tmp_path_factory, 2, 2, 2, 'tiff')
    result = load_local_collection(id=str(sample_dir / 'sample_data.tiff'))
    assert isinstance(result, xr.Dataset)


@pytest.mark.skipif(
    not LocalConnection, reason="environment does not support localprocessing"
)
def test_execute_returns_dataset(tmp_path_factory):
    """LocalConnection.execute() returns a Dataset for a load_collection graph."""
    sample_dir = create_local_data(tmp_path_factory, 2, 2, 2, 'netcdf')
    local_conn = LocalConnection(sample_dir.as_posix())
    flat_graph = {
        "loadcol": {
            "process_id": "load_collection",
            "arguments": {"id": str(sample_dir / "sample_data.nc")},
            "result": True,
        }
    }
    result = local_conn.execute(flat_graph)
    assert isinstance(result, xr.Dataset)
    assert 'temperature' in result.data_vars or 'precipitation' in result.data_vars

def create_local_data(tmp_path_factory,lat_size,lon_size,t_size,file_format):
    np.random.seed(0)
    lon = np.linspace(10.5,11.5,lon_size)
    lat = np.linspace(46.0,47.0,lat_size)
    time = pd.date_range('2014-09-06', periods=t_size)
    reference_time = pd.Timestamp('2014-09-05')

    if file_format.lower() in ['nc','netcdf']:
        temperature = 15 + 8 * np.random.randn(lat_size, lon_size, t_size)
        precipitation = 10 * np.random.rand(lat_size, lon_size, t_size)
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(['x', 'y', 'time'], temperature),
                precipitation=(['x', 'y', 'time'], precipitation),
            ),
            coords=dict(
                lon=(['x'], lon),
                lat=(['y'], lat),
                time=time,
                reference_time=reference_time,
            ),
            attrs=dict(description='Weather related data.'),
        )
        d = tmp_path_factory.mktemp('sample_netcdf')
        ds.to_netcdf(d / 'sample_data.nc')
    elif file_format.lower() in ['tif','tiff','geotiff']:
        temperature = 15 + 8 * np.random.randn(lat_size, lon_size)
        precipitation = 10 * np.random.rand(lat_size, lon_size)
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(['x', 'y'], temperature),
                precipitation=(['x', 'y'], precipitation),
            ),
            coords=dict(
                lon=(['x'], lon),
                lat=(['y'], lat),
            ),
            attrs=dict(description='Weather related data.'),
        )
        d = tmp_path_factory.mktemp('sample_geotiff')
        ds.to_array().transpose('variable', 'y', 'x').rio.to_raster(d / 'sample_data.tiff')

    return d

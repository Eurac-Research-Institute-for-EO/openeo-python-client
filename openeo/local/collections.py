import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import rioxarray
import xarray as xr
from pyproj import Transformer

# type: (Path) -> Optional[dict]
LocalCollectionHandler = Callable[[Path], Optional[dict]]

_LOCAL_COLLECTION_HANDLERS: List[LocalCollectionHandler] = []

_log = logging.getLogger(__name__)

def register_local_collection_handler(handler: LocalCollectionHandler) -> None:
    """
    Register a custom handler for local collections.

    The handler is called with a Path and must either:

      * return a STAC-like collection metadata dict (id, extent, cube:dimensions, ...)
        when it "recognizes" the path, or
      * return None when it does not handle the given path.

    Handlers are tried in registration order. On the first non-None result,
    the returned metadata is added to the local collections list and no
    further handlers are tried for that path.
    """
    _LOCAL_COLLECTION_HANDLERS.append(handler)

def _get_dimension(dims: dict, candidates: List[str]):
    for name in candidates:
        if name in dims:
            return name
    error = f'Dimension matching one of the candidates {candidates} not found! The available ones are {dims}. Please rename the dimension accordingly and try again. This local collection will be skipped.'
    raise Exception(error)


def _get_netcdf_zarr_metadata(file_path):
    if '.zarr' in file_path.suffixes:
        data = xr.open_dataset(file_path.as_posix(),chunks={},engine='zarr')
    else:
        data = xr.open_dataset(file_path.as_posix(),chunks={}) # Add decode_coords='all' if the crs as a band gives some issues
    file_path = file_path.as_posix()
    try:
        t_dim = _get_dimension(data.dims, ['t', 'time', 'temporal', 'DATE'])
    except Exception:
        t_dim = None
    try:
        x_dim = _get_dimension(data.dims, ['x', 'X', 'lon', 'longitude'])
        y_dim = _get_dimension(data.dims, ['y', 'Y', 'lat', 'latitude'])
    except Exception as e:
        _log.warning(e)
        raise Exception(f'Error creating metadata for {file_path}') from e
    metadata = {}
    metadata['stac_version'] = '1.0.0-rc.2'
    metadata['type'] = 'Collection'
    metadata['id'] = file_path
    data_attrs_lowercase = [x.lower() for x in data.attrs]
    data_attrs_original  = [x for x in data.attrs]
    data_attrs = dict(zip(data_attrs_lowercase,data_attrs_original))
    if 'title' in data_attrs_lowercase:
        metadata['title'] = data.attrs[data_attrs['title']]
    else:
        metadata['title'] = file_path
    if 'description' in data_attrs_lowercase:
        metadata['description'] = data.attrs[data_attrs['description']]
    else:
        metadata['description'] = ''
    if 'license' in data_attrs_lowercase:
        metadata['license'] = data.attrs[data_attrs['license']]
    else:
        metadata['license'] = ''
    providers = [{'name':'',
                 'roles':['producer'],
                 'url':''}]
    if 'providers' in data_attrs_lowercase:
        providers[0]['name'] = data.attrs[data_attrs['providers']]
        metadata['providers'] = providers
    elif 'institution' in data_attrs_lowercase:
        providers[0]['name'] = data.attrs[data_attrs['institution']]
        metadata['providers'] = providers
    else:
        metadata['providers'] = providers
    if 'links' in data_attrs_lowercase:
        metadata['links'] = data.attrs[data_attrs['links']]
    else:
        metadata['links'] = ''
    x_min = data[x_dim].min().item(0)
    x_max = data[x_dim].max().item(0)
    y_min = data[y_dim].min().item(0)
    y_max = data[y_dim].max().item(0)

    crs_present = False
    bands = list(data.data_vars)
    if 'crs' in bands:
        bands.remove('crs')
        crs_present = True
    extent = {}
    if crs_present:
        if "crs_wkt" in data.crs.attrs:
            transformer = Transformer.from_crs(data.crs.attrs["crs_wkt"], "epsg:4326")
            lat_min, lon_min = transformer.transform(x_min, y_min)
            lat_max, lon_max = transformer.transform(x_max, y_max)
            extent["spatial"] = {"bbox": [[lon_min, lat_min, lon_max, lat_max]]}

    if t_dim is not None:
        t_min = str(data[t_dim].min().values)
        t_max = str(data[t_dim].max().values)
        extent['temporal'] = {'interval': [[t_min,t_max]]}

    metadata['extent'] = extent

    t_dimension = {}
    if t_dim is not None:
        t_dimension = {t_dim: {'type': 'temporal', 'extent':[t_min,t_max]}}

    x_dimension = {x_dim: {'type': 'spatial','axis':'x','extent':[x_min,x_max]}}
    y_dimension = {y_dim: {'type': 'spatial','axis':'y','extent':[y_min,y_max]}}
    if crs_present:
        if 'crs_wkt' in data.crs.attrs:
            x_dimension[x_dim]['reference_system'] = data.crs.attrs['crs_wkt']
            y_dimension[y_dim]['reference_system'] = data.crs.attrs['crs_wkt']

    b_dimension = {}
    if len(bands)>0:
        b_dimension = {'bands': {'type': 'bands', 'values':bands}}

    metadata['cube:dimensions'] = {**t_dimension,**x_dimension,**y_dimension,**b_dimension}

    return metadata


def _get_geotiff_metadata(file_path):
    data = rioxarray.open_rasterio(file_path.as_posix(),chunks={},band_as_variable=True)
    file_path = file_path.as_posix()
    try:
        t_dim = _get_dimension(data.dims, ['t', 'time', 'temporal', 'DATE'])
    except Exception:
        t_dim = None
    try:
        x_dim = _get_dimension(data.dims, ['x', 'X', 'lon', 'longitude'])
        y_dim = _get_dimension(data.dims, ['y', 'Y', 'lat', 'latitude'])
    except Exception as e:
        _log.warning(e)
        raise Exception(f'Error creating metadata for {file_path}') from e

    metadata = {}
    metadata['stac_version'] = '1.0.0-rc.2'
    metadata['type'] = 'Collection'
    metadata['id'] = file_path
    data_attrs_lowercase = [x.lower() for x in data.attrs]
    data_attrs_original  = [x for x in data.attrs]
    data_attrs = dict(zip(data_attrs_lowercase,data_attrs_original))
    if 'title' in data_attrs_lowercase:
        metadata['title'] = data.attrs[data_attrs['title']]
    else:
        metadata['title'] = file_path
    if 'description' in data_attrs_lowercase:
        metadata['description'] = data.attrs[data_attrs['description']]
    else:
        metadata['description'] = ''
    if 'license' in data_attrs_lowercase:
        metadata['license'] = data.attrs[data_attrs['license']]
    else:
        metadata['license'] = ''
    providers = [{'name':'',
                 'roles':['producer'],
                 'url':''}]
    if 'providers' in data_attrs_lowercase:
        providers[0]['name'] = data.attrs[data_attrs['providers']]
        metadata['providers'] = providers
    elif 'institution' in data_attrs_lowercase:
        providers[0]['name'] = data.attrs[data_attrs['institution']]
        metadata['providers'] = providers
    else:
        metadata['providers'] = providers
    if 'links' in data_attrs_lowercase:
        metadata['links'] = data.attrs[data_attrs['links']]
    else:
        metadata['links'] = ''
    x_min = data[x_dim].min().item(0)
    x_max = data[x_dim].max().item(0)
    y_min = data[y_dim].min().item(0)
    y_max = data[y_dim].max().item(0)

    crs_present = False
    coords = list(data.coords)
    if 'spatial_ref' in coords:
        # bands.remove('crs')
        crs_present = True
    bands = []
    for d in data.data_vars:
        data_attrs_lowercase = [x.lower() for x in data[d].attrs]
        data_attrs_original  = [x for x in data[d].attrs]
        data_attrs = dict(zip(data_attrs_lowercase,data_attrs_original))
        if 'description' in data_attrs_lowercase:
            bands.append(data[d].attrs[data_attrs['description']])
        else:
            bands.append(d)
    extent = {}
    if crs_present:
        if 'crs_wkt' in data.spatial_ref.attrs:
            transformer = Transformer.from_crs(data.spatial_ref.attrs['crs_wkt'], 'epsg:4326')
            lat_min,lon_min = transformer.transform(x_min,y_min)
            lat_max,lon_max = transformer.transform(x_max,y_max)
            extent['spatial'] = {'bbox': [[lon_min, lat_min, lon_max, lat_max]]}

    if t_dim is not None:
        t_min = str(data[t_dim].min().values)
        t_max = str(data[t_dim].max().values)
        extent['temporal'] = {'interval': [[t_min,t_max]]}

    metadata['extent'] = extent

    t_dimension = {}
    if t_dim is not None:
        t_dimension = {t_dim: {'type': 'temporal', 'extent':[t_min,t_max]}}

    x_dimension = {x_dim: {'type': 'spatial','axis':'x','extent':[x_min,x_max]}}
    y_dimension = {y_dim: {'type': 'spatial','axis':'y','extent':[y_min,y_max]}}
    if crs_present:
        if 'crs_wkt' in data.spatial_ref.attrs:
            x_dimension[x_dim]['reference_system'] = data.spatial_ref.attrs['crs_wkt']
            y_dimension[y_dim]['reference_system'] = data.spatial_ref.attrs['crs_wkt']

    b_dimension = {}
    if len(bands)>0:
        b_dimension = {'bands': {'type': 'bands', 'values':bands}}

    metadata['cube:dimensions'] = {**t_dimension,**x_dimension,**y_dimension,**b_dimension}

    return metadata


def _get_local_collections(local_collections_path):
    if isinstance(local_collections_path, str):
        local_collections_path = [local_collections_path]

    local_collections_list = []

    for flds in local_collections_path:
        root = Path(flds)

        # 1) Give registered handlers a first shot at any path in the tree.
        if _LOCAL_COLLECTION_HANDLERS:
            for p in root.rglob("*"):
                # Skip directories/files we already deal with explicitly below
                if p.suffix in [".nc", ".zarr", ".tif", ".tiff"]:
                    continue
                for handler in _LOCAL_COLLECTION_HANDLERS:
                    try:
                        meta = handler(p)
                    except Exception as e:
                        _log.error("Error in local collection handler %r for %s: %s", handler, p, e)
                        meta = None
                    if meta:
                        local_collections_list.append(meta)
                        break  # don't call other handlers for this path

        # 2) Existing NetCDF/Zarr logic (unchanged)
        local_collections_netcdf_zarr = [p for p in root.rglob("*") if p.suffix in [".nc", ".zarr"]]
        for local_file in local_collections_netcdf_zarr:
            try:
                metadata = _get_netcdf_zarr_metadata(local_file)
                local_collections_list.append(metadata)
            except Exception as e:
                _log.error(e)
                continue

        # 3) Existing GeoTIFF logic (unchanged)
        local_collections_geotiffs = [p for p in root.rglob("*") if p.suffix in [".tif", ".tiff"]]
        for local_file in local_collections_geotiffs:
            try:
                metadata = _get_geotiff_metadata(local_file)
                local_collections_list.append(metadata)
            except Exception as e:
                _log.error(e)
                continue

    return {"collections": local_collections_list}

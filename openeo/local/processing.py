import inspect
import logging
from pathlib import Path

import openeo_processes_dask.process_implementations
import openeo_processes_dask.specs
import rioxarray
import xarray as xr
from openeo_pg_parser_networkx import ProcessRegistry
from openeo_pg_parser_networkx.process_registry import Process
from openeo_processes_dask.process_implementations.core import process

_log = logging.getLogger(__name__)


def init_process_registry():
    process_registry = ProcessRegistry(wrap_funcs=[process])

    # Import these pre-defined processes from openeo_processes_dask and register them into registry
    processes_from_module = [
        func
        for _, func in inspect.getmembers(
            openeo_processes_dask.process_implementations,
            inspect.isfunction,
        )
    ]

    specs = {}
    for func in processes_from_module:
        try:
            specs[func.__name__] = getattr(openeo_processes_dask.specs, func.__name__)
        except Exception:
            continue

    for func in processes_from_module:
        try:
            process_registry[func.__name__] = Process(
            spec=specs[func.__name__], implementation=func
            )
        except Exception:
            continue
    return process_registry


PROCESS_REGISTRY = init_process_registry()

# load_stac resolves dynamically through the module to support defair monkey-patch
import openeo_processes_dask.process_implementations.cubes.load as _cubes_load

if "load_stac" in PROCESS_REGISTRY:
    PROCESS_REGISTRY["load_stac"] = Process(
        spec=PROCESS_REGISTRY["load_stac"].spec,
        implementation=lambda *args, **kwargs: _cubes_load.load_stac(*args, **kwargs),
    )


def load_local_collection(*args, **kwargs):
    pretty_args = {k: repr(v)[:80] for k, v in kwargs.items()}
    _log.info("Running process load_collection")
    _log.debug(
            f"Running process load_collection with resolved parameters: {pretty_args}"
        )
    collection = Path(kwargs['id'])
    if '.zarr' in collection.suffixes:
        data = xr.open_dataset(kwargs['id'],chunks={},engine='zarr')
    elif '.nc' in collection.suffixes:
        data = xr.open_dataset(kwargs['id'],chunks={},decode_coords='all') # Add decode_coords='all' if the crs as a band gives some issues
        crs = None
        if 'crs' in data.coords:
            if 'spatial_ref' in data.crs.attrs:
                crs = data.crs.attrs['spatial_ref']
            elif 'crs_wkt' in data.crs.attrs:
                crs = data.crs.attrs['crs_wkt']
        if crs is not None:
            for var in data.data_vars:
                data[var].rio.write_crs(crs, inplace=True)
    elif '.tiff' in collection.suffixes or '.tif' in collection.suffixes:
        data = rioxarray.open_rasterio(kwargs['id'],chunks={},band_as_variable=True)
        for d in list(data.data_vars):
            descriptions = [v for k, v in data[d].attrs.items() if k.lower() == "description"]
            if descriptions:
                data = data.rename({d: descriptions[0]})
    return data

PROCESS_REGISTRY["load_collection"] = Process(
    spec=openeo_processes_dask.specs.load_collection,
    implementation=load_local_collection,
)

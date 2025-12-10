# openeo/local/processing.py

import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import openeo_processes_dask.process_implementations
import openeo_processes_dask.specs
import rioxarray
import xarray as xr
from openeo_pg_parser_networkx import ProcessRegistry
from openeo_pg_parser_networkx.process_registry import Process
from openeo_processes_dask.process_implementations.core import process

_log = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Local load_collection plugin mechanism
# ------------------------------------------------------------------------

# Signature: handler(path: Path, args: Dict[str, Any]) -> xr.Dataset | xr.DataArray | None
_LocalCollectionHandler = Callable[[Path, Dict[str, Any]], Union[xr.Dataset, xr.DataArray, None]]

_LOCAL_COLLECTION_HANDLERS: List[_LocalCollectionHandler] = []


def register_local_collection_handler(handler: _LocalCollectionHandler) -> None:
    """
    Register a custom handler for `load_collection` in local mode.

    The handler is called *before* the built-in NetCDF/Zarr/GeoTIFF logic.
    If it returns an xarray Dataset/DataArray, that result is used.
    If it returns ``None`` or raises, the next handler (or the default logic) is tried.

    Example usage (e.g. for Sentinel-3 OLCI .SEN3 directories)::

        from pathlib import Path
        from openeo.local.processing import register_local_collection_handler

        def s3_olci_handler(path: Path, args: dict):
            if path.suffix.lower() == ".sen3" and path.is_dir():
                # custom OLCI reader, e.g. using xarray + netCDF4
                ds = xr.open_dataset(path / "xfdumanifest.nc")  # placeholder example
                return ds
            return None

        register_local_collection_handler(s3_olci_handler)
    """
    _LOCAL_COLLECTION_HANDLERS.append(handler)


# ------------------------------------------------------------------------
# Process registry initialization
# ------------------------------------------------------------------------


def init_process_registry() -> ProcessRegistry:
    """
    Initialize the ProcessRegistry and populate it with all processes
    from ``openeo_processes_dask.process_implementations``.
    """
    process_registry = ProcessRegistry(wrap_funcs=[process])

    # Import these pre-defined processes from openeo_processes_dask and register them into registry
    processes_from_module = [
        func
        for _, func in inspect.getmembers(
            openeo_processes_dask.process_implementations,
            inspect.isfunction,
        )
    ]

    specs: Dict[str, Any] = {}
    for func in processes_from_module:
        try:
            specs[func.__name__] = getattr(openeo_processes_dask.specs, func.__name__)
        except Exception:
            # Not every function has a spec entry (e.g. internal helpers)
            continue

    for func in processes_from_module:
        try:
            process_registry[func.__name__] = Process(
                spec=specs[func.__name__],
                implementation=func,
            )
        except Exception:
            # Be defensive here: we don't want a single bad process to break the client
            continue
    return process_registry


PROCESS_REGISTRY: ProcessRegistry = init_process_registry()


def register_process(
    process_id: str,
    spec: Any,
    implementation: Callable[..., Any],
    *,
    overwrite: bool = True,
) -> None:
    """
    Convenience helper to extend the local PROCESS_REGISTRY.

    This is especially handy for custom I/O processes or experimental functions
    that should be usable in `LocalConnection.execute`.

    Parameters
    ----------
    process_id:
        The id under which the process is registered
        (e.g. ``"load_collection"`` or ``"load_s3_olci_local"``).
    spec:
        Process spec object, typically something from ``openeo_processes_dask.specs``
        or a spec object compatible with ``openeo_pg_parser_networkx.process_registry.Process``.
    implementation:
        Callable implementing the process.
    overwrite:
        If ``False`` and a process with the same id already exists, a ``KeyError`` is raised.
    """
    if not overwrite and process_id in PROCESS_REGISTRY:
        raise KeyError(f"Process {process_id!r} already present in PROCESS_REGISTRY.")
    PROCESS_REGISTRY[process_id] = Process(spec=spec, implementation=implementation)


# ------------------------------------------------------------------------
# Local load_collection implementation (with plugin hook)
# ------------------------------------------------------------------------


def _load_local_collection_builtin(path: Path, args: Dict[str, Any]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Built-in loader that handles NetCDF, Zarr and GeoTIFF.

    This is called *after* all registered plugin handlers were tried.
    """
    if ".zarr" in path.suffixes:
        # Zarr dataset
        data = xr.open_dataset(str(path), chunks={}, engine="zarr")

    elif ".nc" in path.suffixes:
        # NetCDF dataset
        data = xr.open_dataset(str(path), chunks={}, decode_coords="all")

        # Try to propagate CRS into a `rio`‐compatible attribute
        crs = None
        if "crs" in data.coords:
            if "spatial_ref" in data.crs.attrs:
                crs = data.crs.attrs["spatial_ref"]
            elif "crs_wkt" in data.crs.attrs:
                crs = data.crs.attrs["crs_wkt"]

        data = data.to_array(dim="bands")

        if crs is not None:
            data.rio.write_crs(crs, inplace=True)

    elif any(suffix.lower() in [".tiff", ".tif"] for suffix in path.suffixes):
        # GeoTIFF dataset
        data = rioxarray.open_rasterio(str(path), chunks={}, band_as_variable=True)
        # Normalize variable names using the "description" attribute when present
        for var_name in list(data.data_vars):
            descriptions = [v for k, v in data[var_name].attrs.items() if k.lower() == "description"]
            if descriptions:
                data = data.rename({var_name: descriptions[0]})
        data = data.to_array(dim="bands")

    else:
        raise ValueError(f"Unsupported local collection file type: {path}")

    return data


def load_local_collection(*args, **kwargs):
    """
    Local implementation of the ``load_collection`` process.

    This implementation is registered into ``PROCESS_REGISTRY["load_collection"]``
    so that ``LocalConnection.execute`` can evaluate graphs that use a
    regular ``load_collection`` node with a local path as ``id``.

    Resolution order for handling the collection id:

    1. All plugin handlers registered via :func:`register_local_collection_handler`
       are tried in registration order. First non-None return value wins.
    2. Built-in loader for Zarr, NetCDF and GeoTIFF is used as fallback.

    The function returns an xarray object (Dataset/DataArray) which is then
    handled by the openeo-pg-parser-networkx execution.
    """
    pretty_args = {k: repr(v)[:80] for k, v in kwargs.items()}
    _log.info("Running process load_collection")
    _log.debug(f"Running process load_collection with resolved parameters: {pretty_args}")

    collection = Path(kwargs["id"])

    # 1. Try custom plugin handlers
    for handler in _LOCAL_COLLECTION_HANDLERS:
        try:
            result = handler(collection, kwargs)
        except Exception as e:
            _log.warning(
                "Local collection handler %r failed for %r: %r",
                handler,
                collection,
                e,
            )
            continue
        if result is not None:
            _log.debug("Local collection handler %r handled %r.", handler, collection)
            return result

    # 2. Fallback to built-in logic
    return _load_local_collection_builtin(collection, kwargs)


# Register our local override for load_collection.
# This gives a documented, central way to plug in additional
# local I/O formats (e.g. Sentinel-3 OLCI .SEN3) by using the
# `register_local_collection_handler` helper above.
PROCESS_REGISTRY["load_collection"] = Process(
    spec=openeo_processes_dask.specs.load_collection,
    implementation=load_local_collection,
)

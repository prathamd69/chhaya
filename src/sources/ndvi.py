"""
Sentinel-2 NDVI source.

Downloads the least-cloudy Sentinel-2 scene covering Delhi,
computes NDVI, and zonal-averages it onto the H3 hex grid.

Returns a GeoDataFrame: [h3_index, ndvi]
"""

from pathlib import Path
import hashlib

import geopandas as gpd
import numpy as np
import stackstac
import xarray as xr
from pystac_client import Client
from rasterstats import zonal_stats

from src.sources.base import SourceBase


class Sentinel2Source(SourceBase):
    column_names = ["ndvi"]

    # ---------- public API ----------

    def run(self) -> gpd.GeoDataFrame:
        grid = self._grid()
        scene = self._pick_scene()
        cache = self._cache_path(scene)

        if cache.exists():
            self.log.info(f"using cached NDVI raster: {cache.name}")
            ndvi = xr.open_dataarray(cache).load()
        else:
            red, nir = self._load_bands(scene)
            ndvi = self._compute_ndvi(red, nir)
            self._save_cache(ndvi, cache)

        return self._zonal(ndvi, grid)


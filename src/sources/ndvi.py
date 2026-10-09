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

    # ---------- internals ----------

    def _cfg(self) -> dict:
        return self.cfg["sources"]["sentinel2"]

    def _pick_scene(self):
        cfg = self._cfg()
        self.log.info(f"searching {cfg['collection']} — {cfg['date_range']}")

        catalog = Client.open(cfg["stac_url"])
        items = catalog.search(
            collections=[cfg["collection"]],
            bbox=cfg["bbox"],
            datetime=cfg["date_range"],
            query={"eo:cloud_cover": {"lt": cfg["max_cloud_cover"]}},
        ).item_collection()

        if len(items) == 0:
            raise RuntimeError(
                f"No Sentinel-2 scenes found for {cfg['date_range']} "
                f"with cloud cover < {cfg['max_cloud_cover']}%"
            )

        best = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        self.log.info(
            f"picked {best.id} — "
            f"{best.properties['eo:cloud_cover']:.2f}% cloud — "
            f"{best.datetime.date()}"
        )
        return best

    
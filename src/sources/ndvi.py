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

    def _cache_path(self, scene) -> Path:
            cache_dir = Path(self.cfg["local"]["raw_dir"]) / "sentinel2"
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / f"{scene.id}_ndvi.tif"
    
    def _load_bands(self, scene):
        cfg = self._cfg()
        self.log.info(f"loading bands {cfg['bands']} at {cfg['resolution_m']}m")

        stack = stackstac.stack(
            [scene],
            assets=cfg["bands"],
            epsg=cfg["epsg"],
            resolution=cfg["resolution_m"],
            bounds_latlon=cfg["bbox"],
        ).squeeze()

        red = stack.sel(band="red").compute().astype("float32")
        nir = stack.sel(band="nir").compute().astype("float32")

        self.log.info(f"loaded red shape={red.shape}, nir shape={nir.shape}")
        return red, nir

    def _compute_ndvi(self, red, nir):
        ndvi = (nir - red) / (nir + red)
        valid = (red > 0) & (nir > 0)
        ndvi = ndvi.where(valid)

        self.log.info(
            f"NDVI mean={float(ndvi.mean(skipna=True)):.3f}, "
            f"valid={float(valid.mean()):.1%}"
        )
        return ndvi

    def _save_cache(self, ndvi, path: Path):
        ndvi.rio.to_raster(path)
        self.log.info(f"cached NDVI raster: {path}")

    def _zonal(self, ndvi, grid) -> gpd.GeoDataFrame:
        # Reproject hexes to the raster's CRS (UTM) — cheap, 1,945 polygons
        grid_utm = grid.to_crs(ndvi.rio.crs)

        self.log.info(f"zonal stats over {len(grid_utm)} hexes")

        stats = zonal_stats(
            grid_utm.geometry,
            ndvi.values,
            affine=ndvi.rio.transform(),
            nodata=np.nan,
            stats=["mean"],
        )

        result = gpd.GeoDataFrame(
            {
                "h3_index": grid["h3_index"].values,
                "ndvi": [s["mean"] for s in stats],
            },
            geometry=grid.geometry,
            crs=grid.crs,
        )

        non_null = result["ndvi"].notna().sum()
        self.log.info(f"zonal complete — {non_null}/{len(result)} hexes have NDVI")
        return result
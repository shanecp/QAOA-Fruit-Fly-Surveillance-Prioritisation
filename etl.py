"""Rebuild the locality input from separately obtained original source files."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rebuild(report, geometry, output):
    sources = json.loads((ROOT / 'data/sources.json').read_text())
    for key, path in [('monitoring', report), ('geometry', geometry)]:
        if sha256(path) != sources[key]['sha256']:
            raise ValueError(f'{key} source checksum differs from the study snapshot. Obtain the recorded source version; do not substitute a newer dataset.')
    reader = PdfReader(report)
    pattern = re.compile(r'^\s*(\d{1,2})\s+(\d+(?:\.\d+)?)\s+([A-Z][A-Z ]+?)\s*$')
    rows = []
    for line in reader.pages[4].extract_text(extraction_mode='layout').splitlines():
        match = pattern.match(line)
        if match:
            rows.append({'rank': int(match[1]), 'avg_qff_per_trap': float(match[2]), 'locality_name': match[3].strip()})
    activity = pd.DataFrame(rows)
    if len(reader.pages) != 12 or len(activity) != 39 or activity['rank'].tolist() != list(range(1, 40)) or not activity.locality_name.is_unique:
        raise ValueError('Expected the 39 rows of Table 3 on report page 5')
    geography = gpd.read_file(geometry)
    if len(geography) != 2973 or geography.crs.to_epsg() != 7844:
        raise ValueError('Unexpected geometry source metadata')
    selected = geography[geography.locality_name.isin(activity.locality_name)].sort_values('locality_name').copy()
    if len(selected) != 39 or not selected.locality_name.is_unique:
        raise ValueError('Each monitored locality must match exactly one polygon')
    projected = selected.to_crs(epsg=7855)
    centroids = projected.geometry.centroid
    spatial = pd.DataFrame({'locality_name': projected.locality_name.to_numpy(), 'centroid_x_m': centroids.x.to_numpy(), 'centroid_y_m': centroids.y.to_numpy()})
    anchor = spatial.loc[spatial.locality_name == 'SHEPPARTON'].iloc[0]
    spatial['distance_from_shepparton_km'] = np.sqrt((spatial.centroid_x_m - anchor.centroid_x_m) ** 2 + (spatial.centroid_y_m - anchor.centroid_y_m) ** 2) / 1000.0
    spatial = spatial.sort_values(['distance_from_shepparton_km', 'locality_name'], kind='mergesort')
    spatial['geography_order'] = np.arange(1, 40)
    # Freeze geography-only ordering before activity is joined.
    result = spatial.merge(activity, on='locality_name', validate='one_to_one')
    result = result[['rank', 'locality_name', 'avg_qff_per_trap', 'centroid_x_m', 'centroid_y_m', 'distance_from_shepparton_km', 'geography_order']]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Original March 2026 PDF (obtained separately)')
    parser.add_argument('--geometry', type=Path, required=True, help='Original frozen Vicmap GeoJSON (obtained separately)')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/etl/localities.csv')
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / 'data/processed/localities.csv').resolve():
        parser.error('Write to a new output path; the published input must remain unchanged.')
    print(rebuild(args.report, args.geometry, args.output))


if __name__ == '__main__':
    main()

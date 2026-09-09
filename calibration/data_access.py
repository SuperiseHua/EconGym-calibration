"""Standard-library data interface. Loading collected data never authorizes fitting."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


class CollectedData:
    def __init__(self, bundle):
        self.root = Path(bundle).resolve(strict=True)
        self.contract = self.read('data_contract.json')
        self.manifest = self.read('manifest.json')
        self.sources = {row['id']: row for row in self.manifest['files']}
        if len(self.sources) != len(self.manifest['files']): raise ValueError('duplicate source IDs')
        for item in self.sources.values():
            file = self.resolve(item['bundled_path'])
            if file.stat().st_size != item['bytes'] or sha(file) != item['sha256']:
                raise ValueError(f"source integrity failure: {item['id']}")
        # Prepared data also has a separate frozen-byte manifest, not just raw inputs.
        for item in self.read('prepared_manifest.json')['files']:
            if sha(self.resolve(item['path'])) != item['sha256']:
                raise ValueError(f"prepared data integrity failure: {item['path']}")

    def resolve(self, relative):
        path = (self.root/relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('bundle path escape or non-file')
        return path

    def read(self, relative):
        return json.loads(self.resolve(relative).read_text(encoding='utf-8'))

    def source(self, source_id):
        return self.resolve(self.sources[source_id]['bundled_path'])

    def observations(self, years):
        """Return retrospective candidate observations, not an approved training set."""
        if not years or len(years) != len(set(years)): raise ValueError('unique years required')
        rows = {r['year']: r for r in self.read(self.contract['macro_file'])['rows']}
        if set(years)-rows.keys(): raise ValueError('requested year not prepared; no interpolation')
        return [rows[y] for y in years]

    def calibration_targets(self):
        raise RuntimeError('Real fitting remains closed: approve periods, measurement, full checks, and missing targets first')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', default=str(Path(__file__).resolve().parent/'data/collected_v1'))
    args = p.parse_args()
    data = CollectedData(args.bundle)
    print(json.dumps({'files_verified':len(data.sources), 'data_bundle_available':True,
        'real_calibration_ready':False, 'available_years':[r['year'] for r in data.observations([2021,2022,2023,2024,2025])],
        'taskbook_status':data.contract['taskbook_status']}, ensure_ascii=False, indent=2))

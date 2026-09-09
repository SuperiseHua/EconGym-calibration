"""Taskbook input contracts. Configuration preparation does not run a simulator.

Foundation inputs are external development candidates, never optimizer variables.
Their provenance is retained, not independently certified by this interface.
"""
import copy
import json
import math
from pathlib import Path

FOUNDATION = {'alpha': 'market', 'depreciation_rate': 'bank',
              'initial_capital_output_ratio': 'market'}
UNITS = {'alpha': 'fraction', 'depreciation_rate': 'fraction_per_model_year',
         'initial_capital_output_ratio': 'years'}


def check_foundation(card):
    if card is None:
        return None
    if not isinstance(card, dict) or set(card) != {'schema_version', 'status', 'parameters', 'provenance'}:
        raise ValueError('foundation card requires schema_version, status, parameters, provenance')
    if type(card['schema_version']) is not int or card['schema_version'] != 1:
        raise ValueError('foundation schema_version must be 1')
    if card['status'] != 'development_candidate':
        raise ValueError('only development_candidate foundation cards are supported; no scientific approval is inferred')
    values, provenance = card['parameters'], card['provenance']
    if not isinstance(values, dict) or set(values) != set(FOUNDATION):
        raise ValueError('foundation card must specify exactly the three external foundation parameters')
    if not isinstance(provenance, dict) or set(provenance) != set(FOUNDATION):
        raise ValueError('each foundation parameter requires provenance')
    for key, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key}: finite numeric candidate required; null templates cannot be executed')
        if ((key == 'alpha' and not 0 < value < 1)
                or (key == 'depreciation_rate' and not 0 <= value < 1)
                or (key == 'initial_capital_output_ratio' and value <= 0)):
            raise ValueError(f'{key}: outside structural configuration domain')
        item = provenance[key]
        if not isinstance(item, dict) or set(item) != {'source', 'definition', 'unit', 'year'}:
            raise ValueError(f'{key}: source, definition, unit, year required')
        for field in ('source', 'definition'):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f'{key}: nonempty {field} required')
        if item['unit'] != UNITS[key]:
            raise ValueError(f'{key}: expected unit {UNITS[key]}')
        if type(item['year']) is not int or item['year'] != 2022:
            raise ValueError('this taskbook adapter uses base year 2022 only')
    return copy.deepcopy(card)


def load_foundation(path):
    if path is None:
        return None
    return check_foundation(json.loads(Path(path).read_text(encoding='utf-8')))


def data_readiness(bundle):
    if bundle is None:
        return {'attached': False, 'real_calibration_ready': False}
    from data_access import CollectedData, sha
    data = CollectedData(bundle)
    return {'attached': True, 'bundle': str(data.root), 'source_files_verified': len(data.sources),
            'manifest_sha256': sha(data.resolve('manifest.json')),
            'prepared_manifest_sha256': sha(data.resolve('prepared_manifest.json')),
            'data_contract_sha256': sha(data.resolve('data_contract.json')),
            'real_calibration_ready': False, 'auto_apply_parameters': False,
            'taskbook_status': data.contract['taskbook_status']}


def preparation_status(card, bundle=None):
    foundation = check_foundation(card)
    return {'schema_version': 2, 'interface_prepared': True, 'simulator_called': False,
            'real_calibration_ready': False, 'taskbook_acceptance': 'not_evaluated',
            'foundation_mode': 'explicit_development_candidates' if foundation else 'unestimated_upstream_defaults',
            'foundation_card': foundation, 'data': data_readiness(bundle),
            'blocked_on': ['foundation measurement definitions and candidate approval',
                           'fit/holdout dates, policy path, and data vintage',
                           'headline inflation and fixed-investment mapping',
                           'nominal GDP/unused output definition',
                           'weighted micro contract', 'all sector equations',
                           'agreed search domains and thresholds']}

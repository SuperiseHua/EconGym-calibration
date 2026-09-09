"""Input/configuration regression only. No real simulator or optimizer execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bridge
from taskbook_inputs import check_foundation, load_foundation, preparation_status, UNITS


def fixture_card():
    # Arbitrary software-test values, not estimated economic parameters.
    values = {'alpha': .31, 'depreciation_rate': .07, 'initial_capital_output_ratio': 2.1}
    return {'schema_version': 1, 'status': 'development_candidate', 'parameters': values,
            'provenance': {k: {'source': 'synthetic unit-test fixture; not empirical',
                              'definition': 'configuration-routing test only',
                              'unit': UNITS[k], 'year': 2022} for k in values}}


class FoundationTests(unittest.TestCase):
    def test_absent_card_preserves_upstream(self):
        self.assertIsNone(check_foundation(None))
        cfg = bridge.controlled_config({}, 64, 3, 908701)
        e = {x.entity_name: x.entity_args.params for x in cfg.Environment.Entities}
        self.assertEqual(e['market'].alpha, .36)
        self.assertEqual(e['bank'].depreciation_rate, .06)
        self.assertIsNone(e['market'].initial_capital_output_ratio)

    def test_external_parameters_routed_separately(self):
        card = fixture_card()
        cfg = bridge.controlled_config({'capital_adjustment_speed': .27}, 64, 3, 908701, card)
        e = {x.entity_name: x.entity_args.params for x in cfg.Environment.Entities}
        self.assertEqual(e['market'].alpha, .31)
        self.assertEqual(e['bank'].depreciation_rate, .07)
        self.assertEqual(e['market'].initial_capital_output_ratio, 2.1)
        self.assertEqual(e['bank'].capital_adjustment_speed, .27)
        self.assertEqual(card, fixture_card())

    def test_foundation_cannot_be_optimizer_parameters(self):
        for name, value in fixture_card()['parameters'].items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                bridge.check_parameters({name: value})

    def test_non_dictionary_parameters_rejected(self):
        for value in (None, [], 'alpha'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bridge.check_parameters(value)

    def test_card_is_deep_copied(self):
        original = fixture_card()
        result = check_foundation(original)
        original['parameters']['alpha'] = .9
        original['provenance']['alpha']['source'] = 'changed'
        self.assertEqual(result, fixture_card())

    def test_invalid_values_rejected(self):
        for name, invalid in [('alpha', 0), ('alpha', 1), ('alpha', True),
                              ('alpha', None), ('depreciation_rate', -1),
                              ('depreciation_rate', 1.1), ('initial_capital_output_ratio', 0),
                              ('initial_capital_output_ratio', float('nan'))]:
            with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                card = fixture_card()
                card['parameters'][name] = invalid
                check_foundation(card)

    def test_depreciation_one_rejected_before_model_construction(self):
        # Match entities/bank.py compute_next_kt: 0 <= depreciation_rate < 1.
        card = fixture_card()
        card['parameters']['depreciation_rate'] = 1.0
        with self.assertRaises(ValueError):
            bridge.controlled_config({}, 64, 3, 908701, card)

    def test_missing_or_extra_fields_rejected(self):
        for category in ('parameters', 'provenance'):
            card = fixture_card()
            del card[category]['alpha']
            with self.subTest(category=category), self.assertRaises(ValueError):
                check_foundation(card)
        card = fixture_card()
        card['parameters']['sigma_z'] = .01
        with self.assertRaises(ValueError): check_foundation(card)

    def test_provenance_units_year_required(self):
        for field, invalid in [('source', ''), ('definition', ' '), ('unit', 'percent'),
                              ('year', 2021), ('year', True)]:
            card = fixture_card()
            card['provenance']['alpha'][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError): check_foundation(card)

    def test_status_not_a_scientific_approval(self):
        card = fixture_card()
        card['status'] = 'approved_calibration'
        with self.assertRaises(ValueError): check_foundation(card)

    def test_template_is_non_executable(self):
        with self.assertRaises(ValueError):
            load_foundation(Path(__file__).with_name('foundation_card.template.json'))

    def test_config_does_not_leak_foundation_to_next_call(self):
        bridge.controlled_config({}, 64, 3, 908701, fixture_card())
        cfg = bridge.controlled_config({}, 64, 3, 908701)
        e = {x.entity_name: x.entity_args.params for x in cfg.Environment.Entities}
        self.assertEqual(e['market'].alpha, .36)
        self.assertIsNone(e['market'].initial_capital_output_ratio)

    def test_worker_request_forwards_foundation_without_real_simulator(self):
        request = {'parameters': {}, 'foundation': fixture_card(), 'n': 64,
                   'horizon': 3, 'seeds': [908701]}
        with tempfile.TemporaryDirectory(prefix='agentewm_mock_worker_') as temp:
            with patch.object(bridge, 'run_path', return_value={'valid': False, 'trajectory': []}) as fake:
                response = bridge.worker(request, Path(temp))
            cfg, seed = fake.call_args.args
            e = {x.entity_name: x.entity_args.params for x in cfg.Environment.Entities}
            self.assertEqual(e['market'].alpha, .31)
            self.assertEqual(seed, 908701)
            self.assertFalse(response['valid'])

    def test_worker_rejects_invalid_request_before_simulator(self):
        requests = [
            {'parameters': {}, 'n': 64, 'horizon': 3, 'seeds': [908701, 908701]},
            {'parameters': {}, 'n': 64, 'horizon': 3, 'seeds': [908701], 'unknown': True},
            {'parameters': {}, 'n': 64, 'horizon': 3}]
        with patch.object(bridge, 'run_path') as fake:
            for request in requests:
                with self.subTest(request=request), self.assertRaises(ValueError):
                    bridge.worker(request, Path('not-created'))
            fake.assert_not_called()

    def test_backend_serializes_foundation_to_mock_worker(self):
        with tempfile.TemporaryDirectory(prefix='agentewm_backend_test_') as temp:
            out = Path(temp)/'backend'
            original = fixture_card()
            with patch.object(bridge, 'source_hashes', return_value={'fixture': 'not-a-real-source-hash'}):
                backend = bridge.AgentEWMBackend(Path.cwd(), out, foundation=original)
                original['parameters']['alpha'] = .9
                def fake_process(command, **kwargs):
                    job = Path(command[command.index('--output')+1])
                    request = json.loads((job/'request.json').read_text(encoding='utf-8'))
                    self.assertEqual(request['foundation'], fixture_card())
                    bridge.write_json(job/'response.json', {'valid': False, 'fixture_only': True})
                    return SimpleNamespace(returncode=0, stdout='mock process; no simulator', stderr='')
                with patch.object(bridge.subprocess, 'run', side_effect=fake_process):
                    result = backend.evaluate({'capital_adjustment_speed': .27}, (908701,))
                self.assertTrue(result['fixture_only'])
                self.assertEqual(backend.describe()['foundation'], fixture_card())

    def test_preparation_keeps_real_fitting_closed(self):
        for card in (None, fixture_card()):
            with self.subTest(card=card):
                status = preparation_status(card)
                self.assertFalse(status['real_calibration_ready'])
                self.assertFalse(status['simulator_called'])
                self.assertFalse(status['data']['attached'])

    def test_prepare_cli_with_card_no_simulator(self):
        with tempfile.TemporaryDirectory(prefix='agentewm_prepare_test_') as temp:
            root = Path(temp)
            card_path, request_path = root/'foundation.json', root/'request.json'
            card_path.write_text(json.dumps(fixture_card()), encoding='utf-8')
            request_path.write_text(json.dumps({'parameters': {'price_adjustment_speed': .29}}), encoding='utf-8')
            args = ['launch.py', '--repo', str(Path.cwd()), 'prepare', '--output', str(root/'prepared'),
                    '--foundation-card', str(card_path), '--request', str(request_path)]
            with patch('sys.argv', args), patch.object(bridge, 'run_path', side_effect=AssertionError('no simulation')), \
                    patch.object(bridge, 'AgentEWMBackend', side_effect=AssertionError('no backend')):
                bridge.main()
            result = json.loads((root/'prepared/readiness.json').read_text(encoding='utf-8'))
            self.assertFalse(result['simulator_called'])
            self.assertFalse(result['real_calibration_ready'])
            self.assertEqual(result['foundation_card'], fixture_card())
            self.assertEqual(result['optimizer_parameters'], {'price_adjustment_speed': .29})
            self.assertEqual(set(p.name for p in (root/'prepared').iterdir()),
                             {'controlled_config.yaml', 'readiness.json'})


if __name__ == '__main__':
    unittest.main()

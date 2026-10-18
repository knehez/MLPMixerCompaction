"""Numerical checks for compaction and timing; never download data or start Colab."""
import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    import mlp_mixer_cifar10 as mixer
except ImportError:
    torch = mixer = None


@unittest.skipIf(torch is None, 'Requires project PyTorch/torchvision environment')
class MixerMeasurementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.model = mixer.MLPMixer(depth=2, embedding_dim=16, token_dim=8, channel_dim=32)
        self.masks = []
        for block in self.model.mixer_blocks:
            mask = {}
            for name in ('token_mlp', 'channel_mlp'):
                mlp = getattr(block, name)
                a, b = torch.ones_like(mlp.fc1.weight), torch.ones_like(mlp.fc2.weight)
                a[::2] = 0
                b[:, ::2] = 0
                mask[name + '.fc1'], mask[name + '.fc2'] = a, b
            self.masks.append(mask)

    def test_compaction_preserves_output_and_reduces_actual_parameters(self):
        x = torch.randn(4, 3, 32, 32)
        compact = mixer.compact_and_verify(self.model, self.masks, x)
        self.assertLess(sum(p.numel() for p in compact.parameters()),
                        sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(compact.mixer_blocks[0].token_mlp.fc1.out_features, 4)
        with torch.inference_mode():
            torch.testing.assert_close(compact(x), self.model(x))

    def test_optimizer_cannot_regrow_masked_weights(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        data = torch.randn(4, 3, 32, 32)
        target = torch.tensor([0, 1, 2, 3])
        # Populate optimizer momentum before applying masks.
        mixer.train(self.model, optimizer, [(data, target)], torch.device('cpu'))
        mixer.apply_masks(self.model, self.masks)
        mixer.train(self.model, optimizer, [(data, target)], torch.device('cpu'), self.masks)
        for block, mask in zip(self.model.mixer_blocks, self.masks):
            for name in ('token_mlp', 'channel_mlp'):
                mlp = getattr(block, name)
                self.assertTrue(torch.all(mlp.fc1.weight[mask[name + '.fc1'] == 0] == 0))
                self.assertTrue(torch.all(mlp.fc2.weight[mask[name + '.fc2'] == 0] == 0))
                if mlp.fc1.bias is not None:
                    self.assertTrue(torch.all(mlp.fc1.bias[::2] == 0))

    def test_masked_expansion_preserves_finetuned_weights_and_original_neuron_slots(self):
        for use_bias in (False, True):
            with self.subTest(bias=use_bias):
                import copy
                dense = copy.deepcopy(self.model)
                dims = [(b.token_mlp.fc1.out_features, b.channel_mlp.fc1.out_features) for b in dense.mixer_blocks]
                compact, indices = mixer.prune_to_level(copy.deepcopy(dense), dims, .6, min_hidden=2, return_indices=True)
                if use_bias:
                    # The current Mixer builder uses bias=False; test expansion's
                    # bias copying directly with matching biased linear modules.
                    for network in (dense, compact):
                        for block in network.mixer_blocks:
                            for name in ('token_mlp', 'channel_mlp'):
                                mlp = getattr(block, name)
                                for fc in ('fc1', 'fc2'):
                                    old = getattr(mlp, fc)
                                    new = torch.nn.Linear(old.in_features, old.out_features)
                                    with torch.no_grad():
                                        new.weight.copy_(old.weight)
                                    setattr(mlp, fc, new)
                # Simulate changes made by fine-tuning AFTER neuron selection.
                with torch.no_grad():
                    for parameter in compact.parameters():
                        parameter.add_(torch.randn_like(parameter) * .01)
                masked = mixer.expand_compact_to_masked(compact, dense, indices)
                x = torch.randn(4, 3, 32, 32)
                with torch.inference_mode():
                    torch.testing.assert_close(masked(x), compact(x), rtol=1e-4, atol=1e-5)
                self.assertEqual(sum(p.numel() for p in masked.parameters()), sum(p.numel() for p in dense.parameters()))
                self.assertLess(sum(p.numel() for p in compact.parameters()), sum(p.numel() for p in masked.parameters()))
                for small, large, mapping in zip(compact.mixer_blocks, masked.mixer_blocks, indices):
                    for name in ('token_mlp', 'channel_mlp'):
                        mlp, source = getattr(large, name), getattr(small, name)
                        ids = mapping[name]
                        torch.testing.assert_close(mlp.fc1.weight[ids], source.fc1.weight)
                        removed = [i for i in range(mlp.fc1.out_features) if i not in ids]
                        self.assertTrue(torch.all(mlp.fc1.weight[removed] == 0))
                        self.assertTrue(torch.all(mlp.fc2.weight[:, removed] == 0))

    def test_three_way_benchmark_checks_equivalence_and_exports_compaction_ratio(self):
        import copy
        dims = [(b.token_mlp.fc1.out_features, b.channel_mlp.fc1.out_features) for b in self.model.mixer_blocks]
        compact, indices = mixer.prune_to_level(copy.deepcopy(self.model), dims, .6, min_hidden=2, return_indices=True)
        masked = mixer.expand_compact_to_masked(compact, self.model, indices)
        with tempfile.TemporaryDirectory() as directory:
            result = mixer.benchmark_models(self.model, compact, masked=masked, batch_sizes=(1,),
                                            warmup=0, iterations=1, rounds=3,
                                            output_path=Path(directory) / 'three_way.json')
        self.assertTrue(result['masked_control']['outputs_verified'])
        self.assertEqual(result['parameters']['original'], result['parameters']['masked'])
        batch = result['batches'][0]
        self.assertAlmostEqual(batch['masked_to_pruned_speedup'], batch['masked']['median_ms_per_batch'] /
                               batch['pruned']['median_ms_per_batch'])
        self.assertIn('masked_to_pruned_speedup_batch_1', mixer.masked_csv_fields(result))
        self.assertEqual({mixer.measurement_order({'original': 1, 'pruned': 1, 'masked': 1}, r)[0]
                          for r in range(3)}, {'original', 'pruned', 'masked'})

    def test_benchmark_saves_samples_and_computes_speedup_from_measurements(self):
        compact = mixer.compact_and_verify(self.model, self.masks, torch.randn(2, 3, 32, 32))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'benchmark.json'
            result = mixer.benchmark_models(self.model, compact, batch_sizes=(1, 4),
                                            warmup=1, iterations=2, rounds=3, output_path=output)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertGreater(result['parameter_reduction_fraction'], 0)
            for batch in result['batches']:
                self.assertEqual(len(batch['original']['round_ms_per_batch']), 3)
                self.assertEqual(len(batch['pruned']['round_ms_per_batch']), 3)
                self.assertGreater(batch['pruned']['median_ms_per_batch'], 0)
                self.assertAlmostEqual(batch['speedup'], batch['original']['median_ms_per_batch'] /
                                       batch['pruned']['median_ms_per_batch'])

    def test_mirror_keeps_original_checksum(self):
        with patch.object(mixer.datasets.CIFAR10, '__init__', return_value=None) as base:
            with patch.dict('os.environ', MIXER_CIFAR10_URL='https://mirror.example/cifar.tar.gz'):
                dataset = mixer.MirrorCIFAR10('.', download=True)
            self.assertEqual(dataset.url, 'https://mirror.example/cifar.tar.gz')
            self.assertEqual(dataset.tgz_md5, 'c58f30108f718f92721af3b95e74349a')
            base.assert_called_once_with('.', download=True)

    def test_compiled_pair_warms_up_before_timing_and_uses_requested_mode(self):
        x = torch.randn(1, 3, 32, 32)
        calls = []
        class FakeCompiled:
            def __init__(self, model, name):
                self.model, self.name = model, name
            def __call__(self, value):
                calls.append(('forward', self.name))
                return self.model(value)
        names = iter(('original', 'pruned'))
        def compile_mock(model, **kwargs):
            return FakeCompiled(model, next(names))
        def measure(model, value):
            calls.append(('measure', model.name))
            return (2.0 if model.name == 'original' else 1.0), None
        with patch.object(torch, 'compile', side_effect=compile_mock) as compile_call, \
             patch.object(torch.compiler, 'reset') as reset:
            result = mixer.benchmark_compiled_pair({'original': self.model, 'pruned': self.model},
                                                   x, measure, warmup=2, rounds=2)
        self.assertEqual(compile_call.call_count, 2)
        for call in compile_call.call_args_list:
            self.assertEqual(call.kwargs, {'mode': 'reduce-overhead', 'fullgraph': True, 'dynamic': False})
        first_timing = next(i for i, item in enumerate(calls) if item[0] == 'measure')
        self.assertEqual(first_timing, 6)  # output check plus two warmups per model
        self.assertEqual(calls[first_timing:], [('measure', 'original'), ('measure', 'pruned'),
                                              ('measure', 'pruned'), ('measure', 'original')])
        self.assertEqual(result['speedup'], 2.0)
        self.assertTrue(result['outputs_verified'])
        self.assertEqual(reset.call_count, 2)

    def test_compilation_failure_preserves_eager_result_and_marks_compiled_failed(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(torch, 'compile', side_effect=RuntimeError('compiler unavailable')), \
             patch.object(torch.compiler, 'reset'):
            result = mixer.benchmark_models(self.model, self.model, batch_sizes=(1,), warmup=0,
                                            iterations=1, rounds=1, include_compiled=True,
                                            output_path=Path(directory) / 'result.json')
        self.assertEqual(len(result['batches']), 1)
        self.assertEqual(result['compiled']['status'], 'failed')
        self.assertEqual(result['compiled']['batches'], [])
        self.assertIn('compiler unavailable', result['compiled']['errors'][0]['error'])

    def test_compiled_aggregation_requires_results_from_every_seed(self):
        compiled_batch = {'batch_size': 128, 'speedup': 2.0,
                          'original': {'median_ms_per_batch': 2., 'speedup_vs_eager': 2.},
                          'pruned': {'median_ms_per_batch': 1., 'speedup_vs_eager': 3.}}
        good = {'batches': [{'batch_size': 128}], 'compiled': {'status': 'success', 'batches': [compiled_batch]}}
        bad = {'batches': [{'batch_size': 128}], 'compiled': {'status': 'failed', 'batches': []}}
        complete = mixer.aggregate_compiled_benchmarks([good, good])
        self.assertEqual(complete['status'], 'success')
        self.assertEqual(complete['batches'][0]['pruned']['speedup_vs_eager'], 3.)
        incomplete = mixer.aggregate_compiled_benchmarks([good, bad])
        self.assertEqual(incomplete['status'], 'failed')
        self.assertEqual(incomplete['batches'], [])

    def test_compiled_three_way_control_is_retained_in_aggregation(self):
        batch = {'batch_size': 128, 'speedup': 3., 'masked_to_pruned_speedup': 3.,
                 'original_to_masked_speedup': 1.,
                 'original': {'median_ms_per_batch': 3., 'speedup_vs_eager': 2.},
                 'masked': {'median_ms_per_batch': 3., 'speedup_vs_eager': 2.},
                 'pruned': {'median_ms_per_batch': 1., 'speedup_vs_eager': 4.}}
        benchmark = {'batches': [{'batch_size': 128}], 'compiled': {'status': 'success', 'batches': [batch]}}
        result = mixer.aggregate_compiled_benchmarks([benchmark, benchmark])
        self.assertEqual(result['batches'][0]['masked_to_pruned_speedup'], 3.)
        csv = mixer.compiled_csv_fields({'compiled': result})
        self.assertEqual(csv['compiled_masked_ms_batch_128'], 3.)
        self.assertEqual(csv['compiled_masked_to_pruned_speedup_batch_128'], 3.)

    def test_training_cycle_produces_compact_checkpoint_and_benchmark_without_onnx(self):
        real_model = mixer.MLPMixer
        real_benchmark = mixer.benchmark_models
        dataset = torch.utils.data.TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))

        def tiny_model(**kwargs):
            return real_model(**dict(dict(depth=2, embedding_dim=16, token_dim=8, channel_dim=32), **kwargs))

        def quick_benchmark(original, compact, **kwargs):
            return real_benchmark(original, compact, batch_sizes=(1,), warmup=1, iterations=2, rounds=2, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(mixer, 'MLPMixer', side_effect=tiny_model), \
                     patch.object(mixer, 'MirrorCIFAR10', return_value=dataset), \
                     patch.object(mixer, 'benchmark_models', side_effect=quick_benchmark), \
                     patch.object(torch.cuda, 'is_available', return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    mixer.lottery_ticket_mixer_cycle(mixer.ExperimentConfig(
                        val_size=1, levels=(0.4, 0.75), baseline_epochs=2, finetune_epochs=2,
                        min_epochs=1, patience=1, min_hidden=2, selection_batch=1, depth_removals=()))
                result = json.loads(Path('benchmark_results.json').read_text())
                sweep = json.loads(Path('experiment_results.json').read_text())
                dense = torch.load('original_mixer_model.pt', weights_only=True)
                compact = torch.load(sweep['stages'][-1]['checkpoint'], weights_only=True)
                self.assertEqual(len(sweep['stages']), 3)
                self.assertGreater(sweep['stages'][-1]['actual_reduction'], 0)
                self.assertLess(sum(v.numel() for v in compact.values()), sum(v.numel() for v in dense.values()))
                self.assertIn('accuracy', result)
                self.assertTrue(Path('stage_summary.csv').is_file())
                self.assertEqual(list(Path('.').glob('*.onnx')), [])
            finally:
                os.chdir(old_cwd)

    def test_split_is_reproducible_and_disjoint(self):
        train, val = mixer.split_indices(100, 20, 123)
        self.assertEqual((train, val), mixer.split_indices(100, 20, 123))
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(set(train) | set(val), set(range(100)))

    def test_early_stop_restores_best_weights_instead_of_last_epoch(self):
        metrics = [{'loss': loss, 'accuracy': acc} for loss, acc in
                   ((1.0, .5), (.8, .6), (.85, .59), (.86, .58), (.8, .6))]
        counter = 0
        parameter = next(self.model.parameters())
        def fake_train(*args):
            nonlocal counter
            counter += 1
            with torch.no_grad():
                parameter.fill_(counter)
        config = mixer.ExperimentConfig(baseline_epochs=10, finetune_epochs=10, min_epochs=1, patience=2)
        with patch.object(mixer, 'train', side_effect=fake_train), \
             patch.object(mixer, 'evaluate', side_effect=metrics), contextlib.redirect_stdout(io.StringIO()):
            result = mixer.fit_stage(self.model, None, None, torch.device('cpu'), config, 10, 0.001, 'test')
        self.assertEqual(result['epochs_run'], 3)
        self.assertEqual(result['best_epoch'], 1)
        self.assertTrue(result['stopped_early'])
        self.assertTrue(torch.all(parameter == 1))

    def test_higher_accuracy_checkpoint_wins_even_with_higher_loss(self):
        metrics = [{'loss': loss, 'accuracy': acc} for loss, acc in
                   ((1.0, .5), (.6, .79), (.7, .81), (.7, .81))]
        parameter = next(self.model.parameters())
        counter = 0
        def fake_train(*args):
            nonlocal counter
            counter += 1
            with torch.no_grad():
                parameter.fill_(counter)
        config = mixer.ExperimentConfig(baseline_epochs=2, finetune_epochs=2, min_epochs=1)
        with patch.object(mixer, 'train', side_effect=fake_train), \
             patch.object(mixer, 'evaluate', side_effect=metrics), contextlib.redirect_stdout(io.StringIO()):
            result = mixer.fit_stage(self.model, None, None, torch.device('cpu'), config, 2, .001, 'test')
        self.assertEqual(result['best_epoch'], 2)
        self.assertEqual(result['best_val_accuracy'], .81)
        self.assertTrue(torch.all(parameter == 2))

    def test_aggregation_uses_mean_validation_accuracy_and_reports_seed_variation(self):
        def run(seed, accuracies):
            return {'configuration': {'seed': seed}, 'stages': [
                {'stage': index, 'target_reduction': index * .6, 'parameters': 100 - 60 * index,
                 'actual_reduction': index * .6, 'validation': {'accuracy': acc, 'loss': 1 - acc},
                 'benchmark': {'batches': [{'batch_size': 128, 'original': {'median_ms_per_batch': 10},
                                             'pruned': {'median_ms_per_batch': 10 - index * 4}}]}}
                for index, acc in enumerate(accuracies)]}
        result = mixer.aggregate_repeats([run(123, (.80, .79)), run(456, (.82, .81))], .01, 128)
        self.assertEqual(result['selected_stage'], 1)
        self.assertAlmostEqual(result['stages'][1]['validation']['accuracy'], .8)
        self.assertAlmostEqual(result['stages'][1]['validation_accuracy_std'], .02 / 2 ** .5)
        self.assertAlmostEqual(result['stages'][1]['benchmark']['batches'][0]['speedup'], 10 / 6)

    def test_repeats_keep_split_fixed_and_freeze_selection_before_test(self):
        real_model, real_benchmark = mixer.MLPMixer, mixer.benchmark_models
        data = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 4)
        test_accesses = []
        def dataset(*args, **kwargs):
            if not kwargs.get('train', True):
                frozen = json.loads(Path('repeat_results.json').read_text())
                self.assertEqual(frozen['status'], 'selection_frozen_before_test')
                self.assertEqual(frozen['seeds'], [123, 456])
                test_accesses.append(True)
            return data
        def tiny_model(**kwargs):
            return real_model(**dict(dict(depth=2, embedding_dim=16, token_dim=8, channel_dim=32), **kwargs))
        def quick(original, compact, **kwargs):
            return real_benchmark(original, compact, batch_sizes=(1,), warmup=0, iterations=1, rounds=1, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(mixer, 'MLPMixer', side_effect=tiny_model), \
                     patch.object(mixer, 'MirrorCIFAR10', side_effect=dataset), \
                     patch.object(mixer, 'benchmark_models', side_effect=quick), \
                     patch.object(torch.cuda, 'is_available', return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    summary = mixer.repeat_experiments(mixer.ExperimentConfig(
                        val_size=2, levels=(.60, .85), baseline_epochs=2, finetune_epochs=2,
                        min_epochs=1, patience=1, min_hidden=2, selection_batch=1), (123, 456))
                self.assertEqual(test_accesses, [True])
                self.assertEqual(summary['status'], 'success')
                self.assertEqual(Path('seed_123/split_indices.json').read_bytes(),
                                 Path('seed_456/split_indices.json').read_bytes())
                self.assertTrue(Path('repeat_summary.csv').is_file())
                self.assertEqual(len(summary['held_out_test']), 2)
                for seed in (123, 456):
                    self.assertTrue(Path(f'seed_{seed}/repeat_selected_model.pt').is_file())
                    self.assertTrue(Path(f'seed_{seed}/repeat_selected_masked_model.pt').is_file())
                for test in summary['held_out_test']:
                    self.assertEqual(test['masked']['accuracy'], test['selected']['accuracy'])
            finally:
                os.chdir(old)

    def test_pruning_levels_preserve_width_floor(self):
        dims = [(b.token_mlp.fc1.out_features, b.channel_mlp.fc1.out_features) for b in self.model.mixer_blocks]
        compact = mixer.prune_to_level(self.model, dims, 0.95, min_hidden=4)
        for block in compact.mixer_blocks:
            self.assertGreaterEqual(block.token_mlp.fc1.out_features, 4)
            self.assertGreaterEqual(block.channel_mlp.fc1.out_features, 4)

    def test_block_deletion_equals_bypass_and_respects_depth_floor(self):
        import copy
        candidate = mixer.remove_mixer_block(self.model, 0)
        bypass = copy.deepcopy(self.model).eval()
        bypass.mixer_blocks[0] = torch.nn.Identity()
        x = torch.randn(3, 3, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(candidate(x), bypass(x))
        self.assertEqual(len(candidate.mixer_blocks), 1)
        self.assertEqual(len(self.model.mixer_blocks), 2)
        with self.assertRaises(ValueError):
            mixer.remove_mixer_block(candidate, 0)
        with self.assertRaises(ValueError):
            mixer.remove_mixer_block(self.model, 2)

    def test_block_selection_uses_validation_and_keeps_original_block_ids(self):
        metrics = [{'loss': .4, 'accuracy': .8}, {'loss': .6, 'accuracy': .7}, {'loss': .5, 'accuracy': .79}]
        with patch.object(mixer, 'evaluate', side_effect=metrics) as validation:
            candidate, ids, decision = mixer.choose_block_removal(self.model, [2, 5], None, torch.device('cpu'))
        self.assertEqual(validation.call_count, 3)
        self.assertEqual(ids, [2])
        self.assertEqual(decision['selected']['original_index'], 5)
        self.assertTrue(decision['physical_removal_verified'])
        self.assertEqual(len(candidate.mixer_blocks), 1)

    def test_combined_masked_control_matches_finetuned_shallow_compact_model(self):
        import copy
        dims = [(b.token_mlp.fc1.out_features, b.channel_mlp.fc1.out_features) for b in self.model.mixer_blocks]
        compact, indices = mixer.prune_to_level(copy.deepcopy(self.model), dims, .6, min_hidden=2, return_indices=True)
        compact = mixer.remove_mixer_block(compact, 0)
        with torch.no_grad():
            for parameter in compact.parameters():
                parameter.add_(torch.randn_like(parameter) * .01)
        masked = mixer.expand_compact_to_masked(compact, self.model, indices[1:], kept_blocks=[1])
        x = torch.randn(3, 3, 32, 32)
        restored = mixer.MLPMixer(**mixer.model_config(compact)).eval()
        restored.load_state_dict(compact.state_dict())
        with torch.inference_mode():
            torch.testing.assert_close(masked(x), compact(x), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(restored(x), compact(x))
            torch.testing.assert_close(masked.mixer_blocks[0](x.new_zeros(3, 64, 16)), x.new_zeros(3, 64, 16))
        self.assertEqual(sum(p.numel() for p in masked.parameters()), sum(p.numel() for p in self.model.parameters()))

    def test_full_experiment_includes_separate_depth_and_combined_candidates(self):
        real_model, real_benchmark = mixer.MLPMixer, mixer.benchmark_models
        data = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 4)
        def small(**kwargs):
            return real_model(**dict(dict(depth=2, embedding_dim=16, token_dim=8, channel_dim=32), **kwargs))
        def quick(original, compact, **kwargs):
            return real_benchmark(original, compact, batch_sizes=(1,), warmup=0, iterations=1, rounds=1, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(mixer, 'MLPMixer', side_effect=small), \
                     patch.object(mixer, 'MirrorCIFAR10', return_value=data), \
                     patch.object(mixer, 'benchmark_models', side_effect=quick), \
                     patch.object(torch.cuda, 'is_available', return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = mixer.lottery_ticket_mixer_cycle(mixer.ExperimentConfig(
                        val_size=2, levels=(.6,), combined_width=.6, depth_removals=(1, 2),
                        baseline_epochs=2, finetune_epochs=2, min_epochs=1, patience=1,
                        min_hidden=2, selection_batch=1))
                self.assertEqual([row['family'] for row in result['stages']], ['dense', 'width', 'depth', 'combined'])
                self.assertEqual([row['depth'] for row in result['stages']], [2, 2, 1, 1])
                self.assertLess(result['stages'][-1]['parameters'], result['stages'][1]['parameters'])
                for row in result['stages'][2:]:
                    self.assertEqual(len(row['kept_blocks']), 1)
                    self.assertEqual(len(row['depth_decisions'][0]['candidates']), 2)
                    self.assertTrue(row['benchmark']['masked_control']['outputs_verified'])
                self.assertEqual(result['held_out_test']['masked']['accuracy'], result['held_out_test']['selected']['accuracy'])
            finally:
                os.chdir(old)

    def test_selection_ignores_test_accuracy_and_enforces_validation_tolerance(self):
        def row(stage, acc, latency, params):
            return {'stage': stage, 'validation': {'accuracy': acc}, 'test_accuracy': 1 - acc,
                    'parameters': params, 'benchmark': {'batches': [
                        {'batch_size': 128, 'pruned': {'median_ms_per_batch': latency}}]}}
        stages = [row(0, .80, 10, 100), row(1, .795, 6, 60), row(2, .77, 3, 20)]
        chosen, pareto = mixer.select_candidate(stages, .01, 128)
        self.assertEqual(chosen['stage'], 1)
        self.assertEqual(pareto, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()

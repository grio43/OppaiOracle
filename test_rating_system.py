"""Rating regressions: unknown labels never discard content supervision."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from Configuration_System import load_config
from evaluation_metrics import MetricComputer, ThresholdCalibrator
from loss_functions import AsymmetricFocalLoss, MultiTaskLoss
from test_v2_pipeline import ROOT, make_sidecar
from utils.metadata_ingestion import RATING_TAGS, annotation_tags, rating_to_tag
from vocabulary import TagVocabulary, _count_tags_in_files


def criterion():
    return MultiTaskLoss(AsymmetricFocalLoss(gamma_pos=0, gamma_neg=7, clip=.05, alpha=1.,
                                           label_smoothing=0, ignore_indices=[0, 1]))


class RatingSystemTests(unittest.TestCase):
    def test_rating_formats_keep_every_image_through_arrow_and_workers(self):
        from dataset_loader import SidecarJsonDataset
        import utils.metadata_cache as cache
        known = [(value, RATING_TAGS[i]) for i, code in enumerate('gsqe')
                 for value in (code, code.upper(), RATING_TAGS[i], RATING_TAGS[i][7:], i)]
        known += [(' SAFE ', RATING_TAGS[0]), (' S ', RATING_TAGS[1])]
        unknown = [None, '', ' ', 'unknown', 'UNKNOWN', 'unrated', '?', 'x',
                   -1, 4, 0.0, 1.5, True, False, '0', 'null', [], ['g'], {}, {'rating': 'g'}]
        cases = known + [(value, None) for value in unknown] + [('MISSING_FIELD', None)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for ident, (value, expected) in enumerate(cases):
                path = make_sidecar(root, 'Other', ident, value)
                record = json.loads(path.read_text(encoding='utf-8'))
                # Contradictory inline tokens must not create rating positives.
                record['tags'] += ' rating:general rating:explicit'
                if value == 'MISSING_FIELD':
                    record.pop('rating')
                path.write_text(json.dumps(record), encoding='utf-8')
                self.assertEqual(rating_to_tag(record.get('rating')), expected)
                positives = annotation_tags(record)
                self.assertEqual([t for t in positives if t.startswith('rating:')],
                                 [expected] if expected else [])
                paths.append(path)
            vocab = TagVocabulary(min_frequency=1)
            counts = _count_tags_in_files([str(p) for p in paths], vocab.ignored_tags)
            self.assertEqual(counts['gen:red_hair'], len(cases))
            self.assertEqual(sum(counts[tag] for tag in RATING_TAGS), len(known))
            vocab.build_from_tag_counts(counts, None)
            rating_cols = [vocab.tag_to_index[t] for t in RATING_TAGS]
            with patch.object(cache, '_PROJ_ROOT', root):
                # Uncached, new Arrow cache, then an already-existing Arrow cache.
                for use_cache in (False, True, True):
                    ds = SidecarJsonDataset(root, paths, vocab, image_size=32,
                                            metadata_cache_enabled=use_cache, metadata_cache_workers=1)
                    self.assertEqual(len(ds), len(cases))
                    expected_by_id = {item['image_id']: cases[int(Path(item['filename']).stem)][1]
                                      for item in ds.items}
                    seen = set()
                    for batch in DataLoader(ds, batch_size=8, num_workers=2):
                        self.assertFalse(bool(batch['error'].any()))
                        for image_id, labels in zip(batch['image_id'], batch['tag_labels']):
                            self.assertNotIn(image_id, seen)
                            seen.add(image_id)
                            for content in ('gen:red_hair', 'char:alice', 'artist:alice'):
                                self.assertEqual(labels[vocab.tag_to_index[content]], 1)
                            expected = expected_by_id[image_id]
                            target = (torch.tensor([float(t == expected) for t in RATING_TAGS])
                                      if expected else torch.full((4,), -1.))
                            torch.testing.assert_close(labels[rating_cols].float(), target)
                    self.assertEqual(seen, set(expected_by_id))

    def test_all_unrated_preparation_keeps_all_content_and_split_members(self):
        import dataset_loader
        import utils.metadata_cache as cache
        from utils.v2_preparation import prepare_dataset, verify_preparation
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [make_sidecar(root / 'data', 'Other', i, None) for i in range(9)]
            cfg = load_config(ROOT / 'configs/unified_config.yaml')
            cfg.data.storage_locations = [{'path': str(root / 'data'), 'enabled': True}]
            cfg.data.max_val_samples = 3
            cfg.data.vocab_min_frequency = 1
            cfg.data.preparation_manifest = str(root / 'manifest.json')
            cfg.vocab_path = str(root / 'vocab.json')
            cfg.output_root = str(root / 'experiments')
            with patch.object(dataset_loader, '_PROJ_ROOT', root), patch.object(cache, '_PROJ_ROOT', root):
                manifest = prepare_dataset(cfg)
                self.assertEqual(verify_preparation(cfg), manifest)
                self.assertEqual(manifest['rating_counts'], {'train': {'rated': 0, 'unrated': 6},
                                                            'validation': {'rated': 0, 'unrated': 3}})
                train, val = dataset_loader._try_load_cached_split(root / 'data', cfg.training.seed)
                self.assertEqual(set(train) | set(val), set(paths))
                self.assertFalse(set(train) & set(val))
                vocab = json.loads(Path(cfg.vocab_path).read_text(encoding='utf-8'))
                self.assertEqual(vocab['tag_frequencies']['gen:red_hair'], 9)
                self.assertEqual([vocab['tag_frequencies'][tag] for tag in RATING_TAGS], [0] * 4)

    def test_unrated_batch_has_only_content_loss_and_gradient_under_bf16(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    logits = torch.tensor([[0, 0, -.4, .8, float('nan'), float('inf'), -40, 40],
                                           [0, 0, .6, -.9, .1, 2., -1., 1.]],
                                          device=device, dtype=dtype, requires_grad=True)
                    targets = torch.tensor([[0, 0, 1, 0, -1, -1, -1, -1],
                                            [0, 0, 0, 1, -1, -1, -1, -1]], device=device, dtype=dtype)
                    actual, _ = criterion()(logits, targets)
                    reference, _ = criterion()(logits[:, :4], targets[:, :4])
                    torch.testing.assert_close(actual, reference)
                    expected_grad, = torch.autograd.grad(reference, logits, retain_graph=True)
                    actual.backward()
                    torch.testing.assert_close(logits.grad, expected_grad)
                    self.assertTrue(bool((logits.grad[:, 4:] == 0).all()))
                    self.assertTrue(bool((logits.grad[:, 2:4] != 0).all()))

    def test_sparse_and_unobserved_metrics_are_finite(self):
        p = torch.full((3, 5), .8)
        for average in ('macro', 'micro', 'weighted'):
            computer = MetricComputer(5, mAP_average=average)
            for targets, expected in [([[1, -1, -1, -1, -1]] * 3, 1.),
                                      ([[0, -1, -1, -1, -1]] * 3, 0.),
                                      ([[-1, -1, -1, -1, -1]] * 3, 0.)]:
                t = torch.tensor(targets)
                for metrics in (computer.compute_all_metrics(p, t),
                                computer.compute_all_metrics_at_threshold(p, t, .5)):
                    self.assertEqual(metrics, dict(f1_macro=expected, f1_micro=expected, mAP=expected))
            self.assertEqual(computer.compute_all_metrics(p[:0], torch.empty((0, 5))),
                             dict(f1_macro=0., f1_micro=0., mAP=0.))
        # Micro still penalizes an observed content false positive.
        t = torch.tensor([[1, 0, -1, -1, -1]] * 3)
        self.assertAlmostEqual(MetricComputer(5).compute_all_metrics(p, t)['f1_micro'], 2/3)
        cal = ThresholdCalibrator(mode='per_bucket', default_threshold=.7)
        result = cal.calibrate(p, torch.full_like(p, -1), [str(i) for i in range(5)],
                               frequency_bins=[0, 10], tag_frequencies={str(i): 1 for i in range(5)})
        self.assertTrue(all(value == .7 for value in result.values()))

    def test_masked_metrics_match_independent_known_entry_reference(self):
        import numpy as np
        from sklearn.metrics import average_precision_score, f1_score
        generator = torch.Generator().manual_seed(81)
        p = torch.rand(15, 6, generator=generator)
        t = torch.randint(0, 2, (15, 6), generator=generator)
        t[:, 2:] = -1
        for row in range(0, 15, 3):
            t[row, 2:] = 0
            t[row, 2 + (row // 3) % 4] = 1
        pn, tn = p.numpy(), t.numpy()
        known = tn >= 0
        support = (tn == 1).sum(axis=0)
        aps = np.array([average_precision_score(tn[known[:, c], c], pn[known[:, c], c])
                        for c in range(6)])
        expected_macro_f1 = np.mean([f1_score(tn[known[:, c], c], pn[known[:, c], c] > .5)
                                    for c in range(6) if support[c]])
        for average, expected_ap in [('macro', aps[support > 0].mean()),
                                     ('micro', average_precision_score(tn[known], pn[known])),
                                     ('weighted', np.average(aps, weights=support))]:
            actual = MetricComputer(6, threshold=.5, mAP_average=average).compute_all_metrics(p, t)
            self.assertAlmostEqual(actual['mAP'], expected_ap, places=6)
            self.assertAlmostEqual(actual['f1_macro'], expected_macro_f1, places=6)
            self.assertAlmostEqual(actual['f1_micro'], f1_score(tn[known], pn[known] > .5), places=6)

    def test_telemetry_ignores_unknown_ratings_even_when_rating_metrics_enabled(self):
        from asl_telemetry import ASLDriveManager
        cfg = load_config(ROOT / 'configs/unified_config.yaml')
        cfg.training.asl_telemetry.enabled = True
        cfg.training.asl_telemetry.exclude_rating_tags = False
        cfg.training.asl_telemetry.interval_updates = 1
        cfg.training.asl_telemetry.sibling_groups_path = None
        vocab = TagVocabulary(min_frequency=1)
        vocab.build_from_tag_counts({'gen:solo': 3, 'gen:smile': 2}, None)
        targets = torch.zeros((2, len(vocab)))
        targets[:, vocab.tag_to_index['gen:solo']] = 1
        cols = [vocab.tag_to_index[t] for t in RATING_TAGS]
        targets[:, cols] = -1
        logits = torch.zeros_like(targets)
        changed = logits.clone(); changed[:, cols] = 12
        managers = [ASLDriveManager(cfg, criterion(), vocab, torch.device('cpu'), {}, 0) for _ in range(2)]
        for manager, predictions in zip(managers, (logits, changed)):
            manager.on_update(predictions, targets, global_step=1, epoch0=0)
        for name in ('_dp_mean_ema', '_dp_hard_ema', '_epr_num_ema', '_epr_den_ema'):
            torch.testing.assert_close(getattr(managers[0], name), getattr(managers[1], name))
        self.assertEqual(managers[0].compute_val(logits.sigmoid(), targets, 1, 0),
                         managers[1].compute_val(changed.sigmoid(), targets, 1, 0))

    def test_resume_discards_old_rating_comparisons_but_preserves_progress(self):
        from train_direct import _reconcile_measurement_contract, MEASUREMENT_CONTRACT
        from training_utils import TrainingState
        from schedulers import WarmupStableDecayLR
        opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=.1)
        scheduler = WarmupStableDecayLR(opt, warmup_steps=1, total_steps=30)
        for _ in range(10):
            opt.step(); scheduler.step()
        scheduler.start_cooldown('plateau')
        scheduler.cooldown_best_metric = .99
        state = TrainingState(global_step=10, completed_epochs=2, best_metric=.99, should_stop=True,
                              frozen_macro_tag_indices=[3, 4], eval_history=[{'epoch': 1}],
                              measurement_contract='old', loss_state={'gamma_neg': 7., 'telemetry': {'old': 1}})
        before = copy.deepcopy(scheduler.state_dict())
        self.assertTrue(_reconcile_measurement_contract(state, scheduler))
        self.assertEqual((state.global_step, state.completed_epochs), (10, 2))
        self.assertFalse(state.should_stop)
        self.assertFalse(state.frozen_macro_tag_indices)
        self.assertFalse(state.eval_history)
        self.assertEqual(state.loss_state, {'gamma_neg': 7.})
        self.assertEqual(state.best_metric, -math.inf)
        after = scheduler.state_dict()
        self.assertEqual(after.pop('cooldown_best_metric'), -math.inf)
        before.pop('cooldown_best_metric')
        self.assertEqual(before, after)
        self.assertEqual(state.measurement_contract, MEASUREMENT_CONTRACT)
        self.assertFalse(_reconcile_measurement_contract(state, scheduler))
        # Older checkpoints may omit/null telemetry state entirely.
        state.measurement_contract = 'legacy'
        state.loss_state = None
        self.assertTrue(_reconcile_measurement_contract(state, scheduler))
        self.assertEqual(state.loss_state, {})
        self.assertEqual(state.last_validation_step, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

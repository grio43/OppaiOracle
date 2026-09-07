"""Focused V2 ingestion, preparation, architecture and WSD regressions."""
import copy
import os
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from Configuration_System import load_config
from dataset_loader import SidecarJsonDataset, _map_rating_to_tag
from model_architecture import VisionTransformerConfig, create_model, sample_attention_logits
from schedulers import WarmupStableDecayLR
from utils.metadata_ingestion import parse_tags_field, annotation_tags
from utils.sidecar_discovery import discover_sidecars
from utils.v2_preparation import prepare_dataset, verify_preparation
from vocabulary import TagVocabulary, _count_tags_in_files

ROOT = Path(__file__).resolve().parent


def make_sidecar(root, subset, ident, rating='s'):
    directory = root / subset / 'images' / 'shard_00000'
    directory.mkdir(parents=True, exist_ok=True)
    filename = f'{ident}.jpg'
    Image.new('RGB', (32, 32), 'red').save(directory / filename)
    path = directory / f'{ident}.json'
    path.write_text(json.dumps({'filename': filename, 'rating': rating,
                               'tags': 'gen:red_hair gen:red_hair char:alice artist:alice meta:commentary'}), encoding='utf-8')
    return path


class V2PipelineTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA trainer/optimizer test')
    def test_real_wsd_trainer_finishes_cooldown_and_resumes_completed_run(self):
        import dataset_loader
        import utils.metadata_cache as cache
        import train_direct
        from test_softstop_resume_e2e import build_fixture, build_config
        for unrated_stride in (2, 1):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                build_fixture(root)
                # Test mixed and entirely unrated corpora: every image must reach the
                # real train loop and the complete validation pass, including resume.
                for i, path in enumerate(sorted((root / 'data').glob('*.json'))):
                    if i % unrated_stride == 0:
                        record = json.loads(path.read_text(encoding='utf-8'))
                        record.pop('rating', None)
                        path.write_text(json.dumps(record), encoding='utf-8')
                cfg = build_config(root, {})
                cfg.training.phase = 1
                cfg.training.scheduler = 'wsd'
                cfg.training.warmup_steps = 2
                cfg.training.optimizer = 'adamw'
                cfg.training.eval_steps = 0
                cfg.training.early_stopping_burn_in_epochs = 0
                cfg.model.qk_norm = True
                cfg.model.layer_scale_init = .1
                cfg.threshold_calibration.enabled = False
                cfg.data.max_val_samples = 4
                cfg.data.color_jitter_enabled = False
                cfg.data.random_rotation_enabled = False
                cfg.data.gaussian_blur_enabled = False
                with patch.object(dataset_loader, '_PROJ_ROOT', root), patch.object(cache, '_PROJ_ROOT', root), \
                     patch.dict(os.environ, OO_NON_INTERACTIVE='1', OO_AUTO_REBUILD_VOCAB='0'):
                    train_direct.train_with_orientation_tracking(cfg)
                    last_path = next(root.glob('experiments/**/last.pt'))
                    final = torch.load(last_path, weights_only=False)
                    state = final['scheduler_state_dict']
                    self.assertEqual(final['scheduler_class'], 'WarmupStableDecayLR')
                    self.assertGreaterEqual(state['last_epoch'], state['cooldown_start'] + state['cooldown_steps'])
                    self.assertEqual(state['cooldown_reason'], 'budget')
                    self.assertEqual(state['_last_lr'], [0., 0.])
                    best = torch.load(last_path.parent / 'best_model.pt', weights_only=False)
                    self.assertGreater(best['step'], state['cooldown_start'])
                    before = last_path.stat().st_mtime_ns
                    cfg.training.resume_from = 'latest'
                    train_direct.train_with_orientation_tracking(cfg)
                    self.assertEqual(last_path.stat().st_mtime_ns, before)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA bitsandbytes test')
    def test_head_states_fp32_backbone_dynamic_8bit(self):
        from training_utils import TrainingUtils
        cfg = VisionTransformerConfig(image_size=32, hidden_size=64, num_hidden_layers=1,
                                      num_attention_heads=4, intermediate_size=128, num_tags=1000,
                                      qk_norm=True, layer_scale_init=.1)
        model = create_model(cfg).cuda()
        model.set_onnx_mode(True)
        optimizer = TrainingUtils.get_optimizer(model, 'adamw8bit', 1e-4, fp32_head_optimizer=True)
        model(torch.randn(2, 3, 32, 32, device='cuda'))['logits'].square().mean().backward()
        optimizer.step()
        self.assertEqual(optimizer.state[model.tag_head.weight]['state1'].dtype, torch.float32)
        self.assertEqual(optimizer.state[model.pos_embed]['state1'].dtype, torch.float32)
        state = optimizer.state[model.blocks[0].mlp[0].weight]
        self.assertEqual(state['state1'].dtype, torch.uint8)
        self.assertIn('absmax1', state)
        torch.testing.assert_close(state['qmap1'], optimizer.name2qmap['dynamic'])

    def test_tag_format_rating_and_distinct_image_counts(self):
        self.assertEqual(parse_tags_field('gen:a gen:b'), ['gen:a', 'gen:b'])
        self.assertEqual(parse_tags_field('old tag,other'), ['old tag', 'other'])
        self.assertEqual(parse_tags_field('gen:>,< gen:b'), ['gen:>,<', 'gen:b'])
        for raw, expected in [('g', 'general'), ('s', 'sensitive'), ('q', 'questionable'),
                              ('e', 'explicit'), ('safe', 'general'), (1, 'sensitive')]:
            self.assertEqual(_map_rating_to_tag(raw), f'rating:{expected}')
        self.assertIsNone(_map_rating_to_tag('unknown'))
        with tempfile.TemporaryDirectory() as td:
            path = make_sidecar(Path(td), 'Danbooru', 1)
            vocab = TagVocabulary()
            counts = _count_tags_in_files([str(path)], vocab.ignored_tags)
            self.assertEqual(counts['gen:red_hair'], 1)
            self.assertEqual(counts['rating:sensitive'], 1)
            self.assertNotIn('meta:commentary', counts)
            vocab.build_from_tag_counts(counts, top_k=None)
            self.assertEqual(vocab.tag_frequencies['rating:sensitive'], 1)
            self.assertEqual(len(vocab.get_rating_tag_indices()), 4)

    def test_discovery_preparation_and_subset_invalidation(self):
        import dataset_loader
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / 'data'
            for subset in ('Danbooru', 'Other'):
                for ident in range(8):
                    make_sidecar(data, subset, ident, rating='s' if ident % 2 else None)
            (data / 'Danbooru' / 'updater.json').write_text('{}')
            make_sidecar(data, '.venv', 99)
            self.assertEqual(len(discover_sidecars(data)), 16)
            cfg = load_config(ROOT / 'configs/unified_config.yaml')
            cfg.data.storage_locations = [{'path': str(data), 'enabled': True}]
            cfg.data.max_val_samples = 4
            cfg.data.vocab_min_frequency = 1
            cfg.data.preparation_manifest = str(root / 'state' / 'manifest.json')
            cfg.vocab_path = str(root / 'vocab.json')
            cfg.output_root = str(root / 'experiments')
            with patch.object(dataset_loader, '_PROJ_ROOT', root):
                manifest = prepare_dataset(cfg)
                self.assertEqual(manifest['validation_count'], 4)
                self.assertEqual(manifest['train_count'], 12)
                self.assertEqual(sum(v['unrated'] for v in manifest['rating_counts'].values()), 8)
                saved_vocab = json.loads(Path(cfg.vocab_path).read_text(encoding='utf-8'))
                self.assertEqual(saved_vocab['tag_frequencies']['rating:sensitive'], 8)
                self.assertEqual(verify_preparation(cfg), manifest)
                train, val = dataset_loader._try_load_cached_split(data, cfg.training.seed)
                self.assertFalse(set(train) & set(val))
                make_sidecar(data, 'Third', 1)
                with self.assertRaisesRegex(RuntimeError, 'subsets'):
                    verify_preparation(cfg)
                grown_train, fixed_val = dataset_loader._try_load_cached_split(data, cfg.training.seed)
                self.assertEqual(val, fixed_val)
                self.assertEqual(len(grown_train), 13)

    def test_sensitive_rows_and_subset_ids_agree_in_arrow_and_fallback(self):
        import utils.metadata_cache as cache
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [make_sidecar(root, name, 1, rating=rating) for name, rating in
                     [('Danbooru', 's'), ('Other', None), ('Third', 'unrecognized')]]
            # A conflicting rating token must never override the separate field.
            for path in paths:
                record = json.loads(path.read_text(encoding='utf-8'))
                record['tags'] += ' rating:explicit'
                if record['rating'] is None:
                    record.pop('rating')
                path.write_text(json.dumps(record), encoding='utf-8')
            vocab = TagVocabulary()
            vocab.build_from_tag_counts(_count_tags_in_files([str(p) for p in paths], vocab.ignored_tags), None)
            with patch.object(cache, '_PROJ_ROOT', root):
                ids = []
                for enabled in (False, True):
                    ds = SidecarJsonDataset(root, paths, vocab, image_size=32,
                                            metadata_cache_enabled=enabled, metadata_cache_workers=1)
                    self.assertEqual(len(ds), 3)
                    rows = [ds[i] for i in range(3)]
                    self.assertTrue(all(not row.get('error') for row in rows))
                    for row in rows:
                        labels = row['tag_labels']
                        self.assertEqual(labels[vocab.tag_to_index['gen:red_hair']], 1)
                        ratings = labels[list(vocab.get_rating_tag_indices().values())]
                        self.assertTrue(bool((ratings == -1).all()) or ratings.sum().item() == 1)
                        self.assertNotEqual(labels[vocab.tag_to_index['rating:explicit']], 1)
                    self.assertEqual(sum(bool((r['tag_labels'][list(vocab.get_rating_tag_indices().values())] == -1).all())
                                         for r in rows), 2)
                    row_ids = [row['image_id'] for row in rows]
                    self.assertEqual(len(set(row_ids)), 3)
                    ids.append(row_ids)
                    if enabled:
                        worker = pickle.loads(pickle.dumps(ds.items))
                        self.assertEqual([worker[i]['image_id'] for i in range(3)], row_ids)
                self.assertEqual(*ids)

    def test_unrated_loss_masks_gradients_and_preserves_content_supervision(self):
        from loss_functions import AsymmetricFocalLoss, MultiTaskLoss
        loss_fn = AsymmetricFocalLoss(gamma_pos=0, gamma_neg=7, clip=.05, alpha=1.,
                                     label_smoothing=0, ignore_indices=[0, 1])
        logits = torch.tensor([[.1, .2, -.4, .7, 2., -2., .9, 1.],
                               [.2, .1, .6, -.3, 1., .8, .3, -.4]], requires_grad=True)
        targets = torch.tensor([[0., 0., 1., 0., -1., -1., -1., -1.],
                                [0., 0., 0., 1., 0., 1., 0., 0.]])
        loss, _ = MultiTaskLoss(loss_fn)(logits, targets)
        expected = (loss_fn(logits[:1, :4], targets[:1, :4]) + loss_fn(logits[1:], targets[1:])) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(bool((logits.grad[0, 4:] == 0).all()))
        self.assertTrue(bool((logits.grad[0, 2:4] != 0).all()))
        self.assertTrue(bool((logits.grad[1, 4:] != 0).all()))
        self.assertTrue(bool((logits.grad[:, :2] == 0).all()))
        for reduction in ('none', 'sum', 'mean'):
            loss_fn.reduction = reduction
            self.assertEqual(loss_fn(logits, torch.full_like(targets, -1)).sum().item(), 0)

    def test_unknown_rating_predictions_do_not_change_metrics_or_calibration(self):
        from evaluation_metrics import MetricComputer, FrequencyBucketMetrics, ThresholdCalibrator, ValidationBuffer
        from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision
        from torchmetrics.functional.classification import binary_average_precision
        p = torch.tensor([[.8, .2, .99, .99], [.7, .3, .9, .1], [.1, .9, .4, .8]])
        t = torch.tensor([[1., 0., -1., -1.], [1., 0., 1., 0.], [0., 1., 0., 1.]])
        altered = p.clone(); altered[0, 2:] = 0
        computer = MetricComputer(4, threshold=.5)
        self.assertEqual(computer.compute_all_metrics(p, t), computer.compute_all_metrics(altered, t))
        self.assertEqual(computer.compute_per_tag_metrics(p, t), computer.compute_per_tag_metrics(altered, t))
        for factory in (lambda: MultilabelF1Score(4, average=None, ignore_index=-1),
                        lambda: MultilabelAveragePrecision(4, average=None, thresholds=200, ignore_index=-1)):
            a, b = factory(), factory()
            a.update(p[:2], t[:2].long()); a.update(p[2:], t[2:].long())
            b.update(altered, t.long())
            torch.testing.assert_close(a.compute(), b.compute())
            torch.testing.assert_close(a.compute(), torch.ones(4))
        buckets = FrequencyBucketMetrics({str(i): 5 for i in range(4)}, [0, 10], [str(i) for i in range(4)])
        self.assertEqual(buckets.compute_bucketed_metrics(p, t, .5), buckets.compute_bucketed_metrics(altered, t, .5))
        exact = sum(binary_average_precision(p[t[:, i] >= 0, i], t[t[:, i] >= 0, i].long()).item()
                    for i in range(4)) / 4
        self.assertEqual(buckets.compute_bucketed_metrics(p, t, .5)['0-9']['mAP'], exact)
        for mode in ('per_tag', 'per_bucket'):
            cal = ThresholdCalibrator(mode=mode)
            kwargs = dict(tag_names=[str(i) for i in range(4)], frequency_bins=[0, 10],
                          tag_frequencies={str(i): 5 for i in range(4)})
            self.assertEqual(cal.calibrate(p, t, **kwargs), cal.calibrate(altered, t, **kwargs))
        buffer = ValidationBuffer(3, 4); buffer.append(p, t)
        torch.testing.assert_close(buffer.tensors()[1], t.to(torch.int8))

    def test_rating_priors_use_only_rated_images(self):
        from types import SimpleNamespace
        from model_architecture import initialize_tag_head_bias
        model = SimpleNamespace(tag_head=torch.nn.Linear(2, 2))
        initialize_tag_head_bias(model, {0: 'gen:solo', 1: 'rating:general'},
                                 {'gen:solo': 2, 'rating:general': 2}, 10, rated_samples=4)
        torch.testing.assert_close(model.tag_head.bias.sigmoid(), torch.tensor([.2, .5]))

    def test_wsd_shape_budget_and_exact_resume(self):
        def make():
            p = torch.nn.Parameter(torch.ones(()))
            opt = torch.optim.SGD([p], lr=1.)
            return opt, WarmupStableDecayLR(opt, warmup_steps=2, total_steps=23, cooldown_fraction=.15)
        opt, sched = make()
        self.assertEqual(opt.param_groups[0]['lr'], .5)
        for _ in range(20):
            opt.step(); sched.step()
        self.assertEqual(sched.cooldown_reason, 'budget')
        self.assertEqual(sched.cooldown_start, 20)
        self.assertEqual(sched.cooldown_steps, 3)
        opt2, resumed = make()
        resumed.load_state_dict(copy.deepcopy(sched.state_dict()))
        for _ in range(3):
            opt.step(); sched.step()
            opt2.step(); resumed.step()
            self.assertEqual(sched.get_last_lr(), resumed.get_last_lr())
        self.assertTrue(sched.finished)
        self.assertEqual(sched.get_last_lr(), [0.])
        opt3, changed = make()
        changed.total_steps += 1
        with self.assertRaisesRegex(ValueError, 'geometry'):
            changed.load_state_dict(sched.state_dict())

    def test_wsd_plateau_cooldown_is_persisted(self):
        p = torch.nn.Parameter(torch.ones(()))
        opt = torch.optim.SGD([p], lr=1.)
        sched = WarmupStableDecayLR(opt, warmup_steps=2, total_steps=100)
        for _ in range(20):
            opt.step(); sched.step()
        self.assertTrue(sched.start_cooldown('plateau'))
        self.assertFalse(sched.start_cooldown('plateau'))
        self.assertEqual(sched.cooldown_steps, 3)
        self.assertEqual(sched.state_dict()['cooldown_reason'], 'plateau')

    def test_qk_layerscale_gradients_and_diagnostic_rng(self):
        cfg = VisionTransformerConfig(image_size=32, hidden_size=32, num_hidden_layers=2,
                                      num_attention_heads=4, intermediate_size=48, num_tags=8,
                                      qk_norm=True, layer_scale_init=.1, dropout=0., attention_dropout=0.)
        model = create_model(cfg)
        model.set_onnx_mode(True)
        model.train()
        images = torch.randn(2, 3, 32, 32)
        state = torch.get_rng_state()
        self.assertTrue(0 < sample_attention_logits(model, images) < 1e4)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        outputs = model(images)['logits']
        outputs.square().mean().backward()
        for block in model.blocks:
            self.assertIsNotNone(block.q_norm.weight.grad)
            self.assertTrue(torch.isfinite(block.ls1.grad).all())
            self.assertTrue(torch.equal(block.ls1.detach(), torch.full((32,), .1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

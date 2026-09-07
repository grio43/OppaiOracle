"""Prepared bad-image reporting and bounded binned AP; temporary data only."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchmetrics.classification import MultilabelAveragePrecision
import dataset_loader as dl
import utils.metadata_cache as cache
from Configuration_System import ConfigValidationError, ValidationConfig, load_config
from evaluation_metrics import update_binned_average_precision
from test_v2_pipeline import make_sidecar
from utils.bad_image_report import BadImageReporter
from utils.v2_preparation import prepare_dataset, verify_preparation

ROOT = Path(__file__).resolve().parent


def fixture(root):
    for i in range(64):
        make_sidecar(root / 'data', 'Danbooru' if i % 2 else 'Other', i)
    cfg = load_config(ROOT / 'configs/unified_config.yaml')
    cfg.data.storage_locations = [{'path': str(root / 'data'), 'enabled': True}]
    cfg.data.preparation_manifest = str(root / 'state' / 'manifest.json')
    cfg.data.max_val_samples = 4
    cfg.data.vocab_min_frequency = 1
    cfg.data.metadata_cache_workers = 2
    cfg.data.num_workers = cfg.validation.dataloader.num_workers = 0
    cfg.data.batch_size = cfg.validation.dataloader.batch_size = 4
    cfg.vocab_path = str(root / 'vocab.json')
    cfg.output_root = str(root / 'experiments')
    return cfg


def loaders(cfg):
    return dl.create_dataloaders(cfg.data, cfg.validation, cfg.vocab_path,
                                 cfg.data.storage_locations[0]['path'], seed=cfg.training.seed)


def image_ids(dataset):
    return [row['image_id'] for row in dataset.items]


class ReadinessRegressions(unittest.TestCase):
    def test_preparation_does_not_decode_images_or_build_a_blacklist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = fixture(root)
            next((root / 'data').rglob('*.jpg')).write_bytes(b'not an image')
            with patch.object(dl, '_PROJ_ROOT', root), patch.object(cache, '_PROJ_ROOT', root), \
                 patch('PIL.Image.open', side_effect=AssertionError('no preparation decode scan')):
                manifest = prepare_dataset(cfg)
                self.assertEqual(manifest['train_count'] + manifest['validation_count'], 64)
                self.assertEqual(verify_preparation(cfg), manifest)
                self.assertFalse((root / 'data' / 'cache_exclusions.txt').exists())

    def test_bad_images_keep_resume_positions_and_can_be_repaired_with_arrow_and_fallback(self):
        for enabled in (False, True):
            with self.subTest(arrow=enabled), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg = fixture(root)
                cfg.data.metadata_cache_enabled = enabled
                with patch.object(dl, '_PROJ_ROOT', root), patch.object(cache, '_PROJ_ROOT', root):
                    manifest = prepare_dataset(cfg)
                    train, val, _ = loaders(cfg)
                    before = image_ids(train.dataset)
                    train.sampler.set_epoch(3)
                    sequence = list(train.sampler)
                    offset = 12
                    bad_idx = sequence[offset]
                    bad_id = before[bad_idx]
                    path = train.dataset._get_image_path_for_idx(bad_idx)
                    original = path.read_bytes()
                    path.write_bytes(b'broken while paused')
                    for _ in range(3):
                        self.assertTrue(train.dataset[bad_idx]['error'])
                    self.assertFalse(train.dataset.failed_samples)
                    self.assertFalse(train.dataset.retry_counts)
                    exclusion_file = root / 'data' / 'cache_exclusions.txt'
                    self.assertFalse(exclusion_file.exists())
                    # Legacy exclusions do not shrink or blacklist prepared rows.
                    exclusion_file.write_text(bad_id + '\n')
                    del train, val
                    cfg.data.num_workers = 2
                    cfg.data.prefetch_factor = 1
                    resumed, val, _ = loaders(cfg)
                    self.assertEqual(verify_preparation(cfg), manifest)
                    self.assertEqual(len(resumed.dataset), manifest['train_count'])
                    self.assertEqual(image_ids(resumed.dataset), before)
                    resumed.dataset.set_epoch(3)
                    resumed.sampler.set_epoch(3)
                    resumed.sampler.set_start_index(offset)
                    served, failed = [], []
                    report = BadImageReporter(root / 'bad_images.txt')
                    try:
                        for batch in resumed:
                            served.extend(batch['image_id'])
                            failed.extend(batch['error'].tolist())
                            report.record_batch(batch)
                            report.record_batch(batch)
                    finally:
                        if resumed._iterator is not None:
                            resumed._iterator._shutdown_workers()
                        report.close()
                    self.assertEqual(served, [before[i] for i in sequence[offset:]])
                    self.assertEqual([ident for ident, error in zip(served, failed) if error], [bad_id])
                    lines = report.path.read_text().splitlines()
                    self.assertEqual(len(lines), 1)
                    self.assertEqual(lines[0].split('\t')[0], str(path))
                    self.assertEqual(exclusion_file.read_text(), bad_id + '\n')
                    path.write_bytes(original)
                    self.assertFalse(resumed.dataset[bad_idx]['error'])

    def test_report_deduplicates_across_restarts_without_caller_disk_io(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'bad_images.txt'
            batch = {'error': torch.tensor([True, False, True]),
                     'error_path': [str(Path(td) / 'a' / '1.jpg'), '', str(Path(td) / 'b' / '1.jpg')],
                     'image_id': ['a_1', 'ok', 'b_1'], 'error_reason': ['bad\nimage', '', 'bad\timage']}
            real_open = Path.open
            def background_open(*args, **kwargs):
                self.assertEqual(threading.current_thread().name, 'bad-image-report')
                return real_open(*args, **kwargs)
            for _ in range(2):
                with patch.object(Path, 'open', new=background_open):
                    report = BadImageReporter(path)
                    try:
                        for _ in range(10):
                            report.record_batch(batch)
                    finally:
                        report.close()
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual({line.split('\t')[0] for line in lines}, set(batch['error_path']) - {''})

    def test_chunked_ap_preserves_counts_and_scores_with_unknown_targets(self):
        generator = torch.Generator().manual_seed(41)
        scores = torch.rand(19, 17, generator=generator)
        targets = (torch.rand(19, 17, generator=generator) < .1).long()
        targets[::2, -4:] = -1
        targets[:, -1] = -1
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            p, t = scores.to(device), targets.to(device)
            reference = MultilabelAveragePrecision(17, average=None, thresholds=200, ignore_index=-1).to(device)
            reference.update(p, t)
            for size in (1, 8, 64):
                with self.subTest(device=device, chunk=size):
                    metric = MultilabelAveragePrecision(17, average=None, thresholds=200, ignore_index=-1).to(device)
                    with patch.object(metric, 'update', wraps=metric.update) as calls:
                        update_binned_average_precision(metric, p, t, size)
                        rows = [call.args[0].shape[0] for call in calls.call_args_list]
                        self.assertEqual(sum(rows), len(p))
                        self.assertLessEqual(max(rows), size)
                    self.assertTrue(torch.equal(reference.confmat, metric.confmat))
                    torch.testing.assert_close(reference.compute(), metric.compute(), rtol=0, atol=0, equal_nan=True)

    def test_ap_chunk_size_config_rejects_non_positive_or_non_integer_values(self):
        self.assertEqual(load_config(ROOT / 'configs/unified_config.yaml').validation.ap_update_chunk_size, 8)
        for invalid in (0, -1, True, None, 2.5, '8'):
            cfg = ValidationConfig()
            cfg.ap_update_chunk_size = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(ConfigValidationError, 'ap_update_chunk_size'):
                cfg.validate()

    @unittest.skipUnless(torch.cuda.is_available(), 'real CUDA pause/resume')
    def test_real_wsd_resume_skips_bad_train_and_validation_images(self):
        import train_direct
        from test_softstop_resume_e2e import build_config
        for all_validation_bad in (False, True):
            with self.subTest(all_validation_bad=all_validation_bad), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                prep_cfg = fixture(root)
                cfg = build_config(root, {'batch_size': 4})
                cfg.data.preparation_manifest = prep_cfg.data.preparation_manifest
                cfg.data.max_val_samples = 4
                cfg.data.vocab_min_frequency = 1
                cfg.data.metadata_cache_workers = 2
                cfg.training.phase = 1
                cfg.training.scheduler = 'wsd'
                cfg.training.warmup_steps = 2
                cfg.training.optimizer = 'adamw'
                cfg.training.num_epochs = 2
                cfg.training.gradient_accumulation_steps = 2
                cfg.training.eval_steps = 0
                cfg.training.early_stopping_burn_in_epochs = 0
                cfg.threshold_calibration.enabled = False
                cfg.model.qk_norm = True
                cfg.model.layer_scale_init = .1
                stop_path = Path(cfg.log_dir) / 'STOP_TRAINING'
                real_forward = train_direct.MultiTaskLoss.forward
                requested = False
                def stop_after_first_batch(criterion, *args, **kwargs):
                    nonlocal requested
                    result = real_forward(criterion, *args, **kwargs)
                    if torch.is_grad_enabled() and not requested:
                        stop_path.touch()
                        requested = True
                    return result
                with patch.object(dl, '_PROJ_ROOT', root), patch.object(cache, '_PROJ_ROOT', root), \
                     patch('utils.v2_preparation.verify_phase1_recipe'), \
                     patch.dict(os.environ, OO_NON_INTERACTIVE='1', OO_AUTO_REBUILD_VOCAB='0'):
                    manifest = prepare_dataset(cfg)
                    with patch.object(train_direct.MultiTaskLoss, 'forward', new=stop_after_first_batch):
                        train_direct.train_with_orientation_tracking(cfg)
                    last_path = next(root.glob('experiments/**/last.pt'))
                    paused = torch.load(last_path, weights_only=False)
                    self.assertFalse(paused['training_state']['is_epoch_boundary'])
                    train, val, _ = loaders(cfg)
                    train.sampler.set_epoch(0)
                    bad_idx = list(train.sampler)[paused['training_state']['sample_in_epoch']]
                    bad_paths = [train.dataset._get_image_path_for_idx(bad_idx)]
                    bad_paths += [val.dataset._get_image_path_for_idx(i)
                                  for i in range(len(val.dataset) if all_validation_bad else 1)]
                    for path in bad_paths:
                        path.write_bytes(b'broken while paused')
                    del train, val
                    stop_path.unlink(missing_ok=True)
                    cfg.training.resume_from = 'latest'
                    train_direct.train_with_orientation_tracking(cfg)
                    completed = torch.load(last_path, weights_only=False)
                    self.assertEqual(completed['training_state']['completed_epochs'], 2)
                    self.assertGreater(completed['step'], paused['step'])
                    self.assertEqual(completed['scheduler_state_dict']['_last_lr'], [0., 0.])
                    self.assertEqual(verify_preparation(cfg), manifest)
                    rows = (Path(cfg.log_dir) / 'bad_images.txt').read_text().splitlines()
                    self.assertEqual(len(rows), len(bad_paths))
                    self.assertEqual({row.split('\t')[0] for row in rows}, {str(p) for p in bad_paths})
                    self.assertFalse((root / 'data' / 'cache_exclusions.txt').exists())
                    if all_validation_bad:
                        self.assertFalse(completed['training_state']['eval_history'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

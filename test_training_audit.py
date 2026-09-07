"""Bounded training-audit regressions. All artifacts live in temporary dirs.

Run: L:\\Dab\\payton_env\\Scripts\\python.exe -B test_training_audit.py
"""
import ast
import logging
import os
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import torch
from torchmetrics.functional.classification import (
    multilabel_average_precision, multilabel_f1_score,
)

from Configuration_System import load_config
from asl_telemetry import ASLDriveManager
from dataset_loader import ArrowMetadataAccessor
from evaluation_metrics import FrequencyBucketMetrics, ValidationBuffer
from loss_functions import AsymmetricFocalLoss, MultiTaskLoss
from training_utils import AsyncCheckpointWriter, CheckpointManager, TrainingState

ROOT = Path(__file__).resolve().parent


def criterion(**kw):
    args = dict(gamma_pos=0., gamma_neg=7., alpha=1., clip=.05,
                label_smoothing=0., ignore_indices=[], reduction='sum')
    args.update(kw)
    return AsymmetricFocalLoss(**args)


class AuditRegressions(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'real trainer requires CUDA bf16')
    def test_real_trainer_saves_completion_and_policy_halt(self):
        import dataset_loader
        import utils.metadata_cache as metadata_cache
        import train_direct
        from test_softstop_resume_e2e import build_fixture, build_config
        for halt in (False, True):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                build_fixture(root)
                cfg = build_config(root, {})
                cfg.training.optimizer = 'adamw'
                cfg.training.eval_steps = 1
                cfg.training.early_stopping_mode = 'halt' if halt else 'off'
                cfg.training.num_epochs = 3 if halt else 2
                cfg.training.early_stopping_burn_in_epochs = 0 if halt else 2
                cfg.training.early_stopping_min_epochs = 1
                cfg.training.early_stopping_window = 1
                cfg.training.early_stopping_confirm = 1
                cfg.training.early_stopping_min_delta = 2.
                if halt:
                    cfg.training.early_stopping_threshold = 2.  # Epoch 2 cannot be best.
                cfg.training.early_stopping_phases.phase2 = None
                cfg.training.use_tensorboard = False
                cfg.training.asl_telemetry.enabled = True
                cfg.threshold_calibration.enabled = False
                cfg.data.max_val_samples = 4
                served_ids = {}
                real_factory = train_direct.create_dataloaders
                def observe_loaders(*args, **kwargs):
                    loaders = real_factory(*args, **kwargs)
                    for name, loader in zip(('train', 'eval'), loaders[:2]):
                        items = loader.dataset.items
                        served_ids[name] = {items[i]['image_id'] for i in range(len(items))}
                        worker = pickle.loads(pickle.dumps(items))
                        self.assertEqual(served_ids[name],
                                         {worker[i]['image_id'] for i in range(len(worker))})
                    return loaders
                with patch.object(dataset_loader, '_PROJ_ROOT', root), \
                     patch.object(metadata_cache, '_PROJ_ROOT', root), \
                     patch.object(train_direct, 'create_dataloaders', side_effect=observe_loaders), \
                     patch.dict(os.environ, OO_NON_INTERACTIVE='1', OO_AUTO_REBUILD_VOCAB='0'):
                    train_direct.train_with_orientation_tracking(cfg)
                    train_ids, reserved_ids = dataset_loader._try_load_cached_split(root/'data', seed=cfg.training.seed)
                    self.assertEqual(len(train_ids), 190)
                    self.assertEqual(len(reserved_ids), 10)
                    self.assertEqual(len(served_ids['train']), 196)
                    self.assertEqual(len(served_ids['eval']), 4)
                    self.assertFalse(served_ids['train'] & served_ids['eval'])
                    self.assertTrue({p.stem for p in reserved_ids[4:]} <= served_ids['train'])
                    self.assertEqual(served_ids['eval'], {p.stem for p in reserved_ids[:4]})
                last_path = next(root.glob('experiments/**/last.pt'))
                last = torch.load(last_path, weights_only=False)
                state = last['training_state']
                self.assertEqual(last['epoch'], 2)
                self.assertEqual(state['completed_epochs'], 2)
                self.assertTrue(state['is_epoch_boundary'])
                self.assertEqual(state['batch_in_epoch'], 0)
                self.assertEqual(state['sample_in_epoch'], 0)
                self.assertEqual(len(state['eval_history']), 2)
                self.assertEqual(last['step'], state['global_step'])
                self.assertTrue(last['optimizer_state_dict']['state'])
                self.assertEqual(state['measurement_contract'], train_direct.MEASUREMENT_CONTRACT)
                self.assertEqual(state['loss_state']['gamma_neg'], 7.)
                if halt:
                    self.assertTrue(state['should_stop'])
                    self.assertFalse(last['is_best'])
                    best = torch.load(last_path.parent/'best_model.pt', weights_only=False)
                    self.assertEqual(best['epoch'], 1)
                else:
                    self.assertFalse(last['is_best'])

    def test_confident_negative_gradient_and_detach(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for clip in (.05, .2):
                x = torch.tensor([[0., 2., 6.25, 7.]], dtype=torch.bfloat16,
                                 device=device, requires_grad=True)
                with torch.autocast(device, dtype=torch.bfloat16):
                    loss = criterion(clip=clip)(x, torch.zeros_like(x))
                loss.backward()
                z = x.detach().float().requires_grad_()
                p = z.sigmoid()
                ref = -((p-clip).clamp_min(0).pow(7).detach()
                        * (1-p+clip).clamp_max(1).log()).sum()
                ref.backward()
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.all(x.grad > 0))
                torch.testing.assert_close(x.grad.float(), z.grad, rtol=.005, atol=1e-6)
                torch.testing.assert_close(loss, ref)

    def test_selection_precision_actual_trainer_expression(self):
        tree = ast.parse((ROOT / 'train_direct.py').read_text(encoding='utf-8'))
        expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'probs' for t in n.targets)
                    and "tag_logits" in ast.unparse(n.value))
        logits = torch.tensor([[-1.0703125]*2, [-1.0625]*2], dtype=torch.bfloat16)
        probs = eval(compile(ast.Expression(expr), '<trainer scores>', 'eval'),
                     {'torch': torch, 'outputs': {'tag_logits': logits}})
        targets = torch.tensor([[0, 0], [1, 1]])
        ap = multilabel_average_precision(probs, targets, num_labels=2, thresholds=200)
        self.assertEqual(ap.item(), 1.)

    def test_fixed_loss_rejects_checkpoint_mutation_even_without_telemetry(self):
        cfg = load_config(ROOT / 'configs/unified_config.yaml')
        cfg.training.asl_telemetry.enabled = False
        live = MultiTaskLoss(criterion(reduction='mean', ignore_indices=[0, 1]))
        with self.assertRaisesRegex(ValueError, 'checkpoint gamma_neg=4'):
            ASLDriveManager(cfg, live, None, torch.device('cpu'), {'gamma_neg': 4}, 0)
        self.assertEqual(live.gamma_neg, 7)
        drive = ASLDriveManager(cfg, live, None, torch.device('cpu'), {'gamma_neg': 7}, 0)
        self.assertFalse(drive.request_gamma_step(6., 99))
        self.assertEqual(live.gamma_neg, 7)
        for key, bad in [('detach_focal_weight', False), ('clip', .2)]:
            with self.assertRaisesRegex(ValueError, key):
                ASLDriveManager(cfg, MultiTaskLoss(criterion(**{key: bad})), None,
                                torch.device('cpu'), {}, 0)

    def test_cap_is_the_only_holdout(self):
        tree = ast.parse((ROOT / 'dataset_loader.py').read_text(encoding='utf-8'))
        cap = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                   and ast.unparse(n.test).startswith('max_val_samples and'))
        scope = dict(max_val_samples=30000, val_list=list(range(296056)),
                     train_list=['train'], logger=logging.getLogger('test'))
        exec(compile(ast.Module(body=[cap], type_ignores=[]), '<cap>', 'exec'), scope)
        self.assertEqual(scope['train_list'], ['train'] + list(range(30000, 296056)))
        self.assertEqual(scope['val_list'], list(range(30000)))
        self.assertFalse(set(scope['train_list']) & set(scope['val_list']))

    def test_incompatible_automatic_resume_raises(self):
        tree = ast.parse((ROOT / 'train_direct.py').read_text(encoding='utf-8'))
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
                     and any('Requested resume checkpoint' in ast.unparse(h) for h in n.handlers))
        def incompatible(**kwargs):
            raise ValueError('num_labels mismatch')
        for mode in ('latest', 'best', 'explicit.pt'):
            scope = dict(ckpt_config_preview={}, ckpt_state_dict_keys=[], config=None,
                         validate_config_compatibility=incompatible,
                         resume_opt=mode, ckpt_path=Path('explicit.pt'))
            with self.assertRaisesRegex(ValueError, 'num_labels mismatch'):
                exec(compile(ast.Module(body=[guard], type_ignores=[]), '<resume>', 'exec'), scope)

    def test_worker_mapping_without_selected_table_allocation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'metadata.arrow'
            n = 10000
            table = pa.table(dict(image_id=[str(i) for i in range(n)],
                                  tags=[['tag'*100]]*n, rating=['general']*n,
                                  dir=[td]*n))
            with pa.OSFile(str(path), 'wb') as sink:
                with pa.ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
            with pa.memory_map(str(path), 'r') as source:
                full = pa.ipc.open_file(source).read_all()
                for indices in (np.arange(0, n, 2, dtype=np.uint32),
                                np.arange(1, n, 2, dtype=np.uint32)):
                    parent = ArrowMetadataAccessor(full, path, indices)
                    worker = pickle.loads(pickle.dumps(parent))
                    before = pa.total_allocated_bytes()
                    self.assertEqual(worker[123], parent[123])
                    allocated = pa.total_allocated_bytes() - before
                    self.assertLess(allocated, 65536)
                    self.assertEqual(worker[123]['image_id'], str(indices[123]))
                    self.assertEqual(worker[-1], parent[-1])
                    self.assertEqual(len(worker._table), n)
                    with self.assertRaises(IndexError):
                        worker[len(worker)]
                    del parent, worker
                del full

    def test_bucket_metrics_match_exact_reference_and_empty_singleton(self):
        torch.manual_seed(4)
        p = torch.rand(80, 75)
        t = torch.rand(80, 75) > .85
        t[:, 0] = False
        names = [f't{i}' for i in range(75)]
        metric = FrequencyBucketMetrics({n: 10 for n in names}, [0, 20, 30], names)
        result = metric.compute_bucketed_metrics(p, t, .6)['0-19']
        supported = t.sum(0) > 0
        for name, expected in {
            'f1_micro': multilabel_f1_score(p, t.long(), num_labels=75, average='micro', threshold=.6),
            'f1_macro': multilabel_f1_score(p[:, supported], t[:, supported].long(),
                                           num_labels=74, average='macro', threshold=.6),
            'mAP': multilabel_average_precision(p[:, supported], t[:, supported].long(),
                                               num_labels=74, average='macro'),
        }.items():
            self.assertAlmostEqual(result[name], expected.item(), places=6)
        one = FrequencyBucketMetrics({'a': 1}, [0, 2], ['a'])
        self.assertEqual(one.compute_bucketed_metrics(p[:, :1], t[:, :1])['0-1']['mAP'], 0)
        self.assertEqual(one.compute_bucketed_metrics(torch.tensor([[.9], [.1]]),
                         torch.tensor([[True], [False]]))['0-1']['mAP'], 1)

    def test_validation_buffer_preserves_scores_without_concatenation(self):
        buffer = ValidationBuffer(3, 2)
        p = torch.tensor([[.25534365, .256832]])
        buffer.append(p, torch.tensor([[0, 1]]))
        probs, targs = buffer.tensors()
        torch.testing.assert_close(probs, p, rtol=0, atol=0)
        self.assertEqual(targs.dtype, torch.int8)
        self.assertEqual(probs.data_ptr(), buffer.probabilities.data_ptr())

    def test_queue_backpressure_preserves_last_and_best(self):
        cfg = load_config(ROOT / 'configs/unified_config.yaml')
        cfg.vocab_path = str(ROOT / 'vocabulary.json')  # Existing legacy fixture vocabulary.
        for keep_best in (False, True):
            with tempfile.TemporaryDirectory() as td:
                gate = threading.Event()
                class DelayedWriter(AsyncCheckpointWriter):
                    def _worker(self):
                        gate.wait()
                        super()._worker()
                manager = CheckpointManager(td, keep_best=keep_best, max_checkpoints=1, async_save=False)
                manager._async_writer = DelayedWriter(max_queue_size=2)
                model = torch.nn.Linear(1, 1)
                opt = torch.optim.SGD(model.parameters(), lr=.1)
                errors = []
                def save(step, best):
                    try:
                        manager.save_checkpoint(model, opt, None, 1, step, {'mAP': step/10},
                                                TrainingState(), is_best=best, config=cfg.to_dict())
                    except Exception as exc:
                        errors.append(exc)
                save(1, True)
                save(2, True)
                thread = threading.Thread(target=save, args=(3, not keep_best))
                thread.start()
                # Synchronous fallback must wait, rather than overtake old saves.
                thread.join(.1)
                self.assertTrue(thread.is_alive())
                gate.set()
                thread.join(20)
                self.assertFalse(thread.is_alive())
                manager.shutdown(timeout=20)
                self.assertEqual(errors, [])
                last = torch.load(Path(td)/'last.pt', weights_only=False)
                best = torch.load(Path(td)/'best_model.pt', weights_only=False)
                self.assertEqual(last['step'], 3)
                self.assertEqual(best['step'], 2 if keep_best else 3)
                self.assertLessEqual(len(list(Path(td).glob('checkpoint_*.pt'))), 1)

    def test_async_failure_cannot_report_successful_shutdown(self):
        with tempfile.TemporaryDirectory() as td:
            manager = CheckpointManager(td, async_save=True)
            manager._async_writer._last_error = OSError('disk failure')
            with self.assertRaisesRegex(RuntimeError, 'Async checkpoint save failed'):
                manager.shutdown(timeout=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

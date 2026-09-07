"""Prepare and verify a frozen Phase 1 dataset without starting training."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.metadata_ingestion import annotation_tags, parse_tags_field, rating_to_tag
from utils.sidecar_discovery import discover_sidecars, subset_signature

LOGGER = logging.getLogger(__name__)
VERSION = 2
PREFIXES = ("gen:", "char:", "copyright:", "artist:", "meta:", "rating:")


def _read_sidecar(path):
    raw = path.read_bytes()
    return path, json.loads(raw), hashlib.sha256(raw).hexdigest()


def _read_members(members, workers):
    # Bound queued futures and parsed records; executor.map on six million
    # inputs at once would retain an impractical number of futures on Python 3.11.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(members), 1024):
            yield from executor.map(_read_sidecar, members[start:start + 1024])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def active_root(config):
    locations = [loc for loc in config.data.storage_locations if loc.get('enabled')]
    if len(locations) != 1:
        raise ValueError("V2 expects one dataset container root; put all subsets below it")
    return Path(locations[0]['path']).resolve()


def prepare_dataset(config):
    """Full scan, exact holdout, unique per-image counts, and label snapshots.

    Must be run after all subsets are ready and before training. Existing V2
    checkpoint streams prevent accidental vocabulary/index or holdout changes.
    """
    from dataset_loader import _write_cached_split, _split_cache_paths
    from vocabulary import TagVocabulary, _ignore_hash
    root = active_root(config)
    checkpoints = Path(config.output_root) / config.experiment_name / 'checkpoints'
    if checkpoints.exists() and any(checkpoints.glob('*.pt')):
        raise RuntimeError("V2 checkpoints exist; use a new experiment before rebuilding the frozen dataset")
    destination = Path(config.data.preparation_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed rebuild must never leave an apparently valid old readiness stamp.
    destination.unlink(missing_ok=True)
    files = discover_sidecars(root)
    limit = int(config.data.max_val_samples or 30000)
    if len(files) <= limit:
        raise ValueError(f"Need more than {limit} sidecars for a {limit}-image holdout; found {len(files)}")
    random.Random(config.training.seed).shuffle(files)
    val_files, train_files = files[:limit], files[limit:]
    vocab = TagVocabulary(min_frequency=config.data.vocab_min_frequency)
    counts = Counter()
    snapshots = {}
    label_hashes = {}
    rating_counts = {}
    val_content_ids = set()
    processed = 0
    for split, members in (('validation', val_files), ('train', train_files)):
        rated = 0
        snapshot = destination.parent / f'{split}_labels.jsonl'
        digest = hashlib.sha256()
        with snapshot.open('w', encoding='utf-8', newline='\n') as stream:
            for path, data, source_sha in _read_members(members, config.data.metadata_cache_workers):
                if not isinstance(data, dict):
                    raise ValueError(f'Expected image-sidecar object in {path}')
                tags = parse_tags_field(data.get('tags'))
                if not tags or any(not tag.startswith(PREFIXES) for tag in tags):
                    raise ValueError(f"Missing or non-V2 tags in {path}")
                rated += rating_to_tag(data.get('rating')) is not None
                filename = data.get('filename')
                if not isinstance(filename, str) or Path(filename).name != filename or not (path.parent / filename).is_file():
                    raise ValueError(f"Missing/invalid image filename in {path}")
                content_id = data.get('md5')
                if content_id:
                    if split == 'train' and content_id in val_content_ids:
                        raise ValueError(f"Duplicate image content crosses train/validation: {path}; deduplicate before preparation")
                    if split == 'validation':
                        if content_id in val_content_ids:
                            raise ValueError(f"Duplicate image in validation: {path}; deduplicate before preparation")
                        val_content_ids.add(content_id)
                positives = sorted(t for t in annotation_tags(data) if t not in vocab.ignored_tags)
                counts.update(positives)
                relative = path.relative_to(root).as_posix()
                label_record = json.dumps([relative, positives], ensure_ascii=False, separators=(',', ':'))
                digest.update((label_record + '\n').encode('utf-8'))
                stream.write(json.dumps([relative, source_sha], ensure_ascii=False) + '\n')
                processed += 1
                if processed % 100000 == 0:
                    LOGGER.info('Prepared %s/%s sidecars', processed, len(files))
        snapshots[split] = {'path': str(snapshot.resolve()), 'sha256': file_sha256(snapshot), 'count': len(members)}
        label_hashes[split] = digest.hexdigest()
        rating_counts[split] = {'rated': rated, 'unrated': len(members) - rated}
        LOGGER.info('%s: %s rated, %s unrated images (rating labels masked)',
                    split, rated, len(members) - rated)
    vocab.build_from_tag_counts(counts, top_k=None)
    vocab.save_vocabulary(Path(config.vocab_path))
    from utils.metadata_cache import _arrow_meta_path
    # Labels may have changed without filenames/counts changing. Invalidate only
    # the metadata stamp; the next loader startup rebuilds the Arrow cache.
    _arrow_meta_path(root).unlink(missing_ok=True)
    _write_cached_split(root, train_files, val_files, seed=config.training.seed)
    split_paths = _split_cache_paths(root)
    manifest = {
        'version': VERSION, 'root': str(root), 'subsets': subset_signature(root),
        'seed': config.training.seed, 'max_val_samples': limit,
        'vocab_min_frequency': config.data.vocab_min_frequency,
        'vocab_sha256': file_sha256(config.vocab_path), 'ignore_hash': _ignore_hash(),
        'vocabulary_size': len(vocab), 'train_count': len(train_files), 'validation_count': len(val_files),
        'snapshots': snapshots, 'label_hashes': label_hashes, 'rating_counts': rating_counts,
        'split_files': [{'path': str(p.resolve()), 'sha256': file_sha256(p)} for p in split_paths],
    }
    destination.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def verify_preparation(config):
    """Check frozen inputs cheaply on startup; label hashes retain full scan evidence."""
    from vocabulary import _ignore_hash
    destination = Path(config.data.preparation_manifest)
    if not destination.is_file():
        raise RuntimeError("V2 data is not prepared. After all subsets arrive, run: "
                           "python tools/prepare_v2_dataset.py --config configs/unified_config.yaml")
    manifest = json.loads(destination.read_text(encoding='utf-8'))
    expected = {'version': VERSION, 'root': str(active_root(config)),
                'subsets': subset_signature(active_root(config)), 'seed': config.training.seed,
                'max_val_samples': config.data.max_val_samples,
                'vocab_min_frequency': config.data.vocab_min_frequency,
                'vocab_sha256': file_sha256(config.vocab_path), 'ignore_hash': _ignore_hash()}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"V2 preparation is stale ({key}); prepare again before starting a new run")
    for entry in list(manifest['snapshots'].values()) + manifest['split_files']:
        if file_sha256(entry['path']) != entry['sha256']:
            raise RuntimeError(f"Frozen V2 artifact changed: {entry['path']}")
    return manifest


def verify_phase1_recipe(config):
    """Assert the adopted production shape/precision/schedule before allocation."""
    expected = {
        'model.hidden_size': 896, 'model.num_hidden_layers': 18,
        'model.num_attention_heads': 14, 'model.intermediate_size': 2320,
        'model.patch_size': 16, 'model.hidden_dropout_prob': 0.,
        'model.attention_dropout': 0., 'model.pos_dropout': 0.,
        'model.drop_path_rate': .2, 'model.qk_norm': True, 'model.layer_scale_init': .1,
        'data.image_size': 320, 'training.phase': 1, 'training.scheduler': 'wsd',
        'training.warmup_steps': 10000, 'training.optimizer': 'adamw8bit',
        'training.fp32_head_optimizer': True, 'training.use_amp': True,
        'training.amp_dtype': 'bfloat16', 'training.weight_decay': .05,
        'training.lr_scaling_mode': 'sqrt', 'training.selection_metric': 'val_mAP',
    }
    for path, value in expected.items():
        section, name = path.split('.')
        if getattr(getattr(config, section), name) != value:
            raise ValueError(f'V2 Phase 1 contract requires {path}={value!r}')
    if config.training.num_epochs > 60:
        raise ValueError('V2 Phase 1 is limited to the approved 60 epochs including cooldown')

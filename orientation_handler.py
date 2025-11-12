#!/usr/bin/env python3
"""
Enhanced Orientation Handler for Anime Image Tagger
Handles comprehensive left/right tag swapping for horizontal flip augmentation
"""
from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Any, Optional
from collections import Counter
import warnings
import threading

logger = logging.getLogger(__name__)


# The _paths() function and DEFAULT_CONFIG dictionary that depended on legacy
# configs/paths.yaml have been removed. The OrientationHandler is now initialized
# directly with parameters from the scripts that use it, which in turn will
# be updated to use the unified configuration.


class OrientationHandler:
    """
    Comprehensive handler for orientation-aware tag swapping during horizontal flips.
    Supports explicit mappings, regex patterns, and special handling rules.
    """
    
    def __init__(
        self,
        mapping_file: Optional[Path] = None,
        random_flip_prob: float = 0.0,
        strict_mode: bool = False,  # Changed default to False
        safety_mode: str = "conservative",  # "conservative", "balanced", "permissive"
        skip_unmapped: bool = False
    ):
        """
        Initialize the orientation handler.
        
        Args:
            mapping_file: Path to JSON file with orientation mappings
            random_flip_prob: Probability of random horizontal flips
            strict_mode: If True, fail if flips are enabled but no mapping provided
            safety_mode: How conservative to be about flipping
            skip_unmapped: If True, skip flipping images with unmapped orientation tags
        """
        self.random_flip_prob = random_flip_prob
        self.strict_mode = strict_mode
        self.safety_mode = safety_mode
        self.skip_unmapped = skip_unmapped
        
        # Initialize mappings
        self.explicit_mappings: dict[str, str] = {}
        self.reverse_mappings: dict[str, str] = {}
        self.duplicate_mappings: dict[str, list[str]] = {}
        self.regex_patterns: list[dict[str, Any]] = []
        self.symmetric_tags: set[str] = set()
        self.skip_flip_tags: set[str] = set()
        self.complex_tags: dict[str, Any] = {}
        
        # Track statistics (thread-safe)
        self.stats: dict[str, Any] = {
            'total_flips': 0,
            'skipped_flips': 0,
            'unmapped_tags': set(),
            'unmapped_tag_frequency': {},  # Track how often each unmapped tag appears
            'mapped_tags': set(),
            'blocked_by_text': 0,
            'blocked_by_safety': 0,
            'safe_flips': 0,
            'images_analyzed': 0,
            'unmapped_blocking_frequency': {}  # Track which unmapped tags block flips
        }
        self._stats_lock = threading.Lock()
        
        # Load mappings
        if mapping_file and mapping_file.exists():
            self._load_mappings(mapping_file)
        elif random_flip_prob > 0:
            if strict_mode:
                raise ValueError(
                    f"Horizontal flips enabled (prob={random_flip_prob}) but no orientation mapping provided. "
                    "Please provide orientation_map.json or set random_flip_prob=0"
                )
            else:
                warnings.warn(
                    f"Horizontal flips enabled (prob={random_flip_prob}) but no orientation mapping provided. "
                    "Using minimal default mappings. This may cause incorrect orientation labels!",
                    UserWarning
                )
                self._load_default_mappings()
        else:
            # Flips disabled, no mapping needed
            logger.info("Horizontal flips disabled (random_flip_prob=0)")
    
    def _load_mappings(self, mapping_file: Path) -> None:
        """Load orientation mappings from JSON file.

        Args:
            mapping_file: Path to orientation_map.json

        Raises:
            ValueError: If strict_mode=True and loading fails
        """
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        except FileNotFoundError:
            # Expected - file doesn't exist yet
            msg = (
                f"Orientation mapping file not found: {mapping_file}. "
                f"Using minimal default mappings."
            )
            if self.strict_mode:
                raise ValueError(msg)
            logger.warning(msg)
            self._load_default_mappings()
            return

        except PermissionError as e:
            # Unexpected - can't read file
            msg = (
                f"Permission denied reading orientation mappings: {mapping_file}. "
                f"Check file permissions."
            )
            if self.strict_mode:
                raise ValueError(msg) from e
            logger.error(msg)
            self._load_default_mappings()
            return

        except json.JSONDecodeError as e:
            # Error - file is corrupted
            msg = (
                f"Invalid JSON in orientation mappings file {mapping_file}: {e}. "
                f"File is corrupted or malformed."
            )
            if self.strict_mode:
                raise ValueError(msg) from e
            logger.error(msg)
            logger.error("Using minimal default mappings. Fix the JSON file to use custom mappings.")
            self._load_default_mappings()
            return

        except OSError as e:
            # Error - I/O problem
            msg = f"I/O error reading orientation mappings {mapping_file}: {e}"
            if self.strict_mode:
                raise ValueError(msg) from e
            logger.error(msg)
            self._load_default_mappings()
            return

        # File loaded successfully, now parse it
        try:
            # Load explicit mappings
            self.explicit_mappings = data.get('explicit_mappings', {})
            if not isinstance(self.explicit_mappings, dict):
                raise ValueError(
                    f"'explicit_mappings' must be a dict, got {type(self.explicit_mappings).__name__}"
                )

            # Build reverse map with collision detection (only keep bijective pairs)
            self.reverse_mappings.clear()
            self.duplicate_mappings.clear()
            value_counts: Counter[str] = Counter(self.explicit_mappings.values())
            for k, v in self.explicit_mappings.items():
                if value_counts[v] > 1:
                    self.duplicate_mappings.setdefault(v, []).append(k)
                else:
                    self.reverse_mappings[v] = k

            # Load regex patterns
            patterns_data = data.get('regex_patterns', [])
            if not isinstance(patterns_data, list):
                raise ValueError(
                    f"'regex_patterns' must be a list, got {type(patterns_data).__name__}"
                )

            for pattern_data in patterns_data:
                try:
                    pattern = re.compile(pattern_data['pattern'])
                    self.regex_patterns.append({
                        'pattern': pattern,
                        'replacement': pattern_data['replacement'],
                        'description': pattern_data.get('description', '')
                    })
                except re.error as e:
                    logger.error(
                        f"Invalid regex pattern '{pattern_data.get('pattern', '')}': {e}. "
                        f"Skipping this pattern."
                    )
                except KeyError as e:
                    logger.error(
                        f"Regex pattern missing required field {e}: {pattern_data}. "
                        f"Skipping this pattern."
                    )

            # Load symmetric tags (don't need swapping)
            symmetric_data = data.get('symmetric_tags', [])
            if not isinstance(symmetric_data, list):
                raise ValueError(
                    f"'symmetric_tags' must be a list, got {type(symmetric_data).__name__}"
                )
            self.symmetric_tags = set(symmetric_data)

            # Load tags that should skip flipping
            skip_data = data.get('skip_flip_tags', [])
            if not isinstance(skip_data, list):
                raise ValueError(
                    f"'skip_flip_tags' must be a list, got {type(skip_data).__name__}"
                )
            self.skip_flip_tags = set(skip_data)

            # Load complex asymmetric tags
            complex_data = data.get('complex_asymmetric_tags', {})
            if not isinstance(complex_data, dict):
                raise ValueError(
                    f"'complex_asymmetric_tags' must be a dict, got {type(complex_data).__name__}"
                )
            self.complex_tags = complex_data

            logger.info(
                f"✓ Loaded orientation mappings from {mapping_file}: "
                f"{len(self.explicit_mappings)} explicit, "
                f"{len(self.regex_patterns)} regex patterns, "
                f"{len(self.symmetric_tags)} symmetric, "
                f"{len(self.skip_flip_tags)} skip tags"
            )

        except (ValueError, TypeError, KeyError) as e:
            # Error - valid JSON but wrong structure
            msg = (
                f"Invalid structure in orientation mappings file {mapping_file}: {e}. "
                f"File has correct JSON syntax but wrong field types or missing fields."
            )
            if self.strict_mode:
                raise ValueError(msg) from e
            logger.error(msg)
            logger.error("Using minimal default mappings. Fix the file structure to use custom mappings.")
            self._load_default_mappings()
            return

        except Exception as e:
            # Unexpected error - bug in code or truly unexpected condition
            msg = f"Unexpected error loading orientation mappings from {mapping_file}: {e}"
            logger.error(msg, exc_info=True)
            if self.strict_mode:
                raise ValueError(msg) from e
            logger.error("Using minimal default mappings.")
            self._load_default_mappings()
            return
    
    def _load_default_mappings(self):
        """Load minimal default mappings as fallback."""
        self.explicit_mappings = {
            'hair_over_left_eye': 'hair_over_right_eye',
            'hair_over_right_eye': 'hair_over_left_eye',
            'left_eye_closed': 'right_eye_closed',
            'right_eye_closed': 'left_eye_closed',
            'hand_on_left_hip': 'hand_on_right_hip',
            'hand_on_right_hip': 'hand_on_left_hip',
            'looking_to_the_left': 'looking_to_the_right',
            'looking_to_the_right': 'looking_to_the_left',
            'facing_left': 'facing_right',
            'facing_right': 'facing_left'
        }
        self.reverse_mappings = {v: k for k, v in self.explicit_mappings.items()}
        
        # Add basic regex patterns
        self.regex_patterns = [
            {
                'pattern': re.compile(r'(.*)_left_(.*)'),
                'replacement': r'\1_right_\2',
                'description': 'Generic left to right'
            },
            {
                'pattern': re.compile(r'(.*)_right_(.*)'),
                'replacement': r'\1_left_\2',
                'description': 'Generic right to left'
            }
        ]
        
        logger.warning("Using minimal default orientation mappings")

    def can_safely_flip(self, tags: list[str]) -> tuple[bool, list[str]]:
        """
        Determine if an image can be safely flipped using conservative logic.
       
        Args:
            tags: List of tags for the image
            
        Returns:
            Tuple of (can_flip, list_of_reasons)
        """
        reasons = []
        
        # Track that we analyzed this image
        with self._stats_lock:
            self.stats['images_analyzed'] += 1
        
        # 1. NEVER flip if text/watermarks present
        if any(tag in self.skip_flip_tags for tag in tags):
            with self._stats_lock:
                self.stats['blocked_by_text'] += 1
            return False, ["Contains text/watermark/signature"]
        
        # 2. Check safety mode
        if self.safety_mode != "permissive":
            # Determine orientation‑sensitive tags
            orientation_indicators = ['left', 'right', 'facing', 'looking', 'pointing',
                                     'turned', 'profile', 'view_from']

            orientation_tags = []
            for tag in tags:
                if any(indicator in tag.lower() for indicator in orientation_indicators):
                    if any(skip in tag for skip in ['copyright', 'bright', 'upright',
                                                     'straight', 'light', 'fight']):
                        continue
                    if any(skip in tag for skip in ['asymmetric', 'asymmetrical', 'single_']):
                        continue
                    if tag in self.symmetric_tags:
                        continue
                    orientation_tags.append(tag)

            if not orientation_tags:
                with self._stats_lock:
                    self.stats['safe_flips'] += 1
                return True, ["No orientation-specific tags found"]

            unmapped = []
            if self.safety_mode == "conservative":
                # Require explicit mapping for all orientation tags
                for tag in orientation_tags:
                    if tag not in self.explicit_mappings and tag not in self.reverse_mappings:
                        unmapped.append(tag)
            else:  # balanced
                for tag in orientation_tags:
                    swapped = self.swap_tag(tag)
                    if swapped == tag:
                        unmapped.append(tag)

            if unmapped:
                # In balanced mode, allow flips to proceed even if some
                # orientation-sensitive tags are unmapped – unless the user
                # explicitly requested to skip when unmapped.
                if self.safety_mode == "balanced" and not self.skip_unmapped:
                    with self._stats_lock:
                        self.stats['safe_flips'] += 1
                    return True, ["Unmapped orientation tags present"]
                with self._stats_lock:
                    self.stats['blocked_by_safety'] += 1
                    for tag in unmapped:
                        self.stats['unmapped_blocking_frequency'][tag] = \
                            self.stats['unmapped_blocking_frequency'].get(tag, 0) + 1
                return False, [f"Unmapped orientation tags: {unmapped[:3]}..."]

        # Permissive mode or all checks passed
        with self._stats_lock:
            self.stats['safe_flips'] += 1
        return True, ["All orientation tags have mappings"]
    
    def should_skip_flip(self, tags: list[str]) -> bool:
        """
        Check if image should skip flipping based on tags.
        
        Args:
            tags: List of tags for the image
            
        Returns:
            True if flip should be skipped
        """
        # Use the new safety logic
        can_flip, reasons = self.can_safely_flip(tags)
        
        if not can_flip:
            with self._stats_lock:
                self.stats['skipped_flips'] += 1
            logger.debug(f"Skipping flip: {reasons[0]}")
            return True
        
        return False
    
    def swap_tag(self, tag: str, *, record_stats: bool = True) -> str:
        """
        Swap a single tag for horizontal flip.

        Args:
            tag: Original tag

        Returns:
            Swapped tag or original if no swap needed
        """
        # Check if symmetric (no swap needed)
        if tag in self.symmetric_tags:
            return tag

        # Check explicit mappings first (highest priority)
        if tag in self.explicit_mappings:
            if record_stats:
                with self._stats_lock:
                    self.stats['mapped_tags'].add(tag)
            return self.explicit_mappings[tag]

        # Check reverse mappings
        if tag in self.reverse_mappings:
            if record_stats:
                with self._stats_lock:
                    self.stats['mapped_tags'].add(tag)
            return self.reverse_mappings[tag]

        # Try regex patterns
        for regex_data in self.regex_patterns:
            pattern = regex_data['pattern']
            if pattern.match(tag):
                try:
                    swapped = pattern.sub(regex_data['replacement'], tag)
                    if swapped != tag:  # Only count if actually changed
                        if record_stats:
                            with self._stats_lock:
                                self.stats['mapped_tags'].add(tag)
                        return swapped
                except Exception as e:
                    logger.error(f"Regex substitution failed for tag '{tag}': {e}")

        # No mapping found - check if it's an orientation tag we should track
        # Note: 'asymmetric'/'asymmetrical' tags are NOT tracked as unmapped since they don't need swapping
        if any(keyword in tag for keyword in ['left', 'right']) and not any(skip in tag for skip in ['asymmetric', 'asymmetrical']):
            # Mirror can_safely_flip() filters fully:
            #  • ignore benign substrings (incl. 'copyright')
            #  • ignore single_* descriptors
            if not (any(sub in tag for sub in ['bright', 'upright', 'straight', 'light', 'fight', 'copyright'])
                    or ('single_' in tag)):
                if record_stats:
                    with self._stats_lock:
                        self.stats['unmapped_tags'].add(tag)
                        self.stats['unmapped_tag_frequency'][tag] = self.stats['unmapped_tag_frequency'].get(tag, 0) + 1

        return tag
    
    def _swap_tag_unlocked(self, tag: str) -> str:
        """Internal version of swap_tag that doesn't acquire locks. For use when lock is already held."""
        return self.swap_tag(tag)
    
    def swap_tags(self, tags: list[str], *, skip_safety_check: bool = False, record_stats: bool = True) -> tuple[list[str], bool]:
        """
        Swap all orientation-sensitive tags in a list atomically.

        CR-017: This method guarantees atomic tag swapping by building a complete
        new list of swapped tags before returning. The caller receives either:
        - The complete swapped list (all tags processed), OR
        - The original list unchanged (if flip was skipped)
        Never a partial swap, even in multithreaded scenarios.

        Args:
            tags: Original list of tags (not modified in-place)
            skip_safety_check: Skip safety checks if already validated upstream
            record_stats: Whether to record statistics (default True)

        Returns:
            Tuple of (swapped_tags, flip_applied):
                - swapped_tags: New list with all tags swapped (or original if skipped)
                - flip_applied: True if flip was applied, False if skipped

        Thread Safety:
            Multiple threads can safely call this method concurrently.
            Stats updates are protected by lock, tag swapping is lock-free.
        """
        # Optionally respect the safety decision upstream to avoid double-veto and stat drift
        if not skip_safety_check and self.should_skip_flip(tags):
            return tags, False

        # Pre-compile filter check for efficiency (avoids duplicating logic from swap_tag)
        def is_unmapped_orientation(tag: str) -> bool:
            return (
                any(keyword in tag for keyword in ['left', 'right']) and
                not any(skip in tag for skip in ['asymmetric', 'asymmetrical', 'bright',
                                                  'upright', 'straight', 'light', 'fight',
                                                  'copyright', 'single_'])
            )

        # CR-017: Build complete swapped list before returning (atomic from caller perspective)
        # Swap tags and collect stats in single pass (avoids second iteration)
        swapped_tags: list[str] = []  # New list - original 'tags' never modified
        mapped_tags = []
        unmapped_tags = []

        for tag in tags:
            swapped = self.swap_tag(tag, record_stats=False)
            swapped_tags.append(swapped)

            if swapped != tag:
                # Tag was mapped
                mapped_tags.append(tag)
            elif is_unmapped_orientation(tag):
                # Tag is unmapped orientation
                unmapped_tags.append(tag)

        # CR-017: Validate atomicity - ensure we processed all tags
        assert len(swapped_tags) == len(tags), \
            f"Tag swap incomplete: {len(swapped_tags)} swapped vs {len(tags)} original"

        # Batch stats update - single lock acquisition with minimal work under lock
        if record_stats:
            with self._stats_lock:
                self.stats['total_flips'] += 1

                # Update mapped tags
                self.stats['mapped_tags'].update(mapped_tags)

                # Update unmapped tags
                self.stats['unmapped_tags'].update(unmapped_tags)
                for tag in unmapped_tags:
                    self.stats['unmapped_tag_frequency'][tag] = \
                        self.stats['unmapped_tag_frequency'].get(tag, 0) + 1

        return swapped_tags, True
    
    def precompute_all_mappings(self, known_vocabulary: set[str]) -> dict[str, str]:
        """Pre-compute all possible tag swaps for a known vocabulary.
        
        Args:
            known_vocabulary: Set of all known tags
            
        Returns:
            Dictionary mapping each orientation tag to its flipped version
        """
        full_mapping = {}
        
        for tag in known_vocabulary:
            # Skip if already computed
            if tag in full_mapping:
                continue
                
            swapped = self.swap_tag(tag)
            if swapped != tag:
                full_mapping[tag] = swapped
                # Add reverse mapping too
                full_mapping[swapped] = tag
        
        logger.info(f"Pre-computed {len(full_mapping)} orientation mappings from vocabulary of {len(known_vocabulary)} tags")
        return full_mapping
    
    def can_swap_eye_colors(self, eye_color_tags: list[str]) -> bool:
        """Check if eye color tags can be properly swapped for heterochromia."""
        # This is a simplified check - expand based on your tag format
        left_colors = [t for t in eye_color_tags if 'left' in t]
        right_colors = [t for t in eye_color_tags if 'right' in t]
        
        # Can only swap if we have both left and right eye color tags
        return len(left_colors) > 0 and len(right_colors) > 0
    
    def handle_complex_tags(self, tags: list[str]) -> tuple[list[str], bool]:
        """Handle complex asymmetric cases like heterochromia.
        
        Args:
            tags: List of tags for the image
            
        Returns:
            Tuple of (possibly modified tags, whether flip should proceed)
        """
        # Check for heterochromia with specific eye colors
        if 'heterochromia' in tags:
            # Look for patterns like "blue_left_eye" and "green_right_eye"
            eye_color_tags = [t for t in tags if 'eye' in t and any(
                color in t for color in ['blue', 'green', 'red', 'yellow', 'purple', 'amber', 'violet']
            )]
            
            if eye_color_tags and not self.can_swap_eye_colors(eye_color_tags):
                # Skip flip if we can't properly swap eye colors
                logger.debug(f"Skipping flip due to complex heterochromia tags: {eye_color_tags}")
                with self._stats_lock:
                    self.stats['skipped_flips'] += 1
                return tags, False
        
        # Check for other complex asymmetric tags
        for tag in tags:
            if tag in self.complex_tags:
                tag_info = self.complex_tags[tag]
                if tag_info.get('requires_special_handling', False):
                    # For now, skip these unless we have specific handlers
                    logger.debug(f"Skipping flip due to complex tag requiring special handling: {tag}")
                    with self._stats_lock:
                        self.stats['skipped_flips'] += 1
                    return tags, False
        
        # Proceed with normal tag swapping
        return self.swap_tags(tags)
    
    def validate_dataset_tags(self, all_tags: set[str], *, record_stats: bool = False) -> dict[str, list[str]]:
        """Validate that all orientation-specific tags in dataset have mappings.
        
        Args:
            all_tags: Set of all unique tags in the dataset
            
        Returns:
            Dictionary of validation issues found
        """
        issues: dict[str, list[str]] = {
            'unmapped_orientation_tags': [],
            'bidirectional_mismatch': []
        }
        
        # CHANGED: Only validate tags that actually need orientation mapping
        # Skip asymmetric and single_ as they don't need mapping
        orientation_indicators = ['left', 'right', 'facing', 'looking', 'pointing']
        skip_indicators = ['asymmetric', 'asymmetrical', 'single_', 'copyright', 'bright', 'upright']
        
        for tag in all_tags:
            # Check if this might be an orientation tag
            if any(indicator in tag for indicator in orientation_indicators):
                # Skip false positives and style descriptors
                if any(skip in tag for skip in skip_indicators):
                    continue                
                # Skip if it's known to be symmetric
                if tag in self.symmetric_tags:
                    continue
                    
                # Check if this tag has a proper mapping
                swapped = self.swap_tag(tag, record_stats=record_stats)
                if swapped == tag:
                    issues['unmapped_orientation_tags'].append(tag)
                else:
                    # Verify bidirectionality
                    double_swapped = self.swap_tag(swapped, record_stats=record_stats)
                    if double_swapped != tag:
                        issues['bidirectional_mismatch'].append(f"{tag} -> {swapped} -> {double_swapped}")
        
        # Sort for consistent output
        for key in issues:
            issues[key] = sorted(issues[key])
        
        # Only return non-empty issues
        return {k: v for k, v in issues.items() if v}
    
    def validate_mappings(self) -> dict[str, list[str]]:
        """
        Validate the consistency of orientation mappings.
        
        Returns:
            Dictionary of validation issues found
        """
        issues: dict[str, list[str]] = {
            # Note: 'bidirectional_mapping_issues' checks A->B->A consistency, not style asymmetry
            'bidirectional_mapping_issues': [],  # Renamed from 'asymmetric_mappings' for clarity
            'circular_mappings': [],
            'duplicate_mappings': [],
            'regex_conflicts': []
        }
        
        # Check for bidirectional mapping consistency (A->B but B doesn't map back to A)
        for tag1, tag2 in self.explicit_mappings.items():
            if tag2 in self.explicit_mappings:
                if self.explicit_mappings[tag2] != tag1:
                    issues['bidirectional_mapping_issues'].append(f"{tag1} -> {tag2} -> {self.explicit_mappings[tag2]}")
        
        # Check for circular mappings
        visited = set()
        for start_tag in self.explicit_mappings:
            if start_tag in visited:
                continue
            
            chain = [start_tag]
            current = self.explicit_mappings.get(start_tag)
            
            while current and current not in chain:
                chain.append(current)
                current = self.explicit_mappings.get(current)
            
            if current in chain and len(chain) > 2:
                issues['circular_mappings'].append(" -> ".join(chain + [current]))
            
            visited.update(chain)
        
        # Duplicate targets (non-bijective explicit mappings)
        value_to_keys: dict[str, list[str]] = {}
        for k, v in self.explicit_mappings.items():
            value_to_keys.setdefault(v, []).append(k)
        for v, keys in value_to_keys.items():
            if len(keys) > 1:
                issues['duplicate_mappings'].append(f"Value '{v}' mapped from multiple keys: {sorted(keys)}")

        # Check for tags that match multiple regex patterns
        test_tags = list(self.explicit_mappings.keys()) + ['left_hand', 'right_foot', 'asymmetrical_wings']
        for tag in test_tags:
            matches = []
            for i, regex_data in enumerate(self.regex_patterns):
                if regex_data['pattern'].match(tag):
                    matches.append(i)

            if len(matches) > 1:
                patterns = [self.regex_patterns[i]['description'] for i in matches]
                issues['regex_conflicts'].append(f"{tag} matches patterns: {patterns}")

        return {k: v for k, v in issues.items() if v}
    
    def get_usage_statistics(self) -> dict[str, Any]:
        """Get usage statistics."""
        with self._stats_lock:
            total_flips = self.stats['total_flips']
            skipped_flips = self.stats['skipped_flips']
            num_mapped = len(self.stats['mapped_tags'])
            num_unmapped = len(self.stats['unmapped_tags'])
            unmapped_sample = list(self.stats['unmapped_tags'])[:20]
            # Sort unmapped tags by frequency
            unmapped_by_freq = sorted(
                self.stats['unmapped_tag_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]            
        attempted = total_flips + skipped_flips
        return {
            'total_flips': total_flips,
            'skipped_flips': skipped_flips,
            'attempted_flips': attempted,
            'skip_rate': skipped_flips / max(1, attempted),
            'num_mapped_tags': num_mapped,
            'num_unmapped_tags': num_unmapped,
            'unmapped_tags_sample': unmapped_sample,
            'unmapped_tags_by_frequency': unmapped_by_freq,
            'flip_safety_stats': {
                'blocked_by_text': self.stats.get('blocked_by_text', 0),
                'blocked_by_safety': self.stats.get('blocked_by_safety', 0),
                'safe_flips': self.stats.get('safe_flips', 0),
                'images_analyzed': self.stats.get('images_analyzed', 0)
            }
        }

    def get_statistics(self) -> dict[str, Any]:
        """Alias for get_usage_statistics for backwards compatibility."""
        return self.get_usage_statistics()
    
    def suggest_mappings(self, unmapped_tags: Optional[set[str]] = None) -> dict[str, str]:
        """
        Auto-suggest potential mappings for unmapped tags.
        
        Args:
            unmapped_tags: Set of tags to analyze, or None to use all tracked unmapped tags
            
        Returns:
            Dictionary of suggested mappings
        """
        if unmapped_tags is None:
            with self._stats_lock:
                unmapped_tags = self.stats['unmapped_tags'].copy()
        
        suggestions: dict[str, str] = {}
        
        for tag in unmapped_tags:
            # Try simple left/right swapping
            if 'left' in tag:
                suggested = tag.replace('left', 'right')
                suggestions[tag] = suggested
            elif 'right' in tag:
                suggested = tag.replace('right', 'left')
                suggestions[tag] = suggested
            # Try pattern-based suggestions
            elif tag.startswith('single_'):
                # Single items don't need swapping (they're already handled correctly)
                suggestions[tag] = tag  # Map to itself (symmetric)
            # Note: asymmetric/asymmetrical tags are NOT included here since they're not tracked as unmapped
        
        return suggestions
    
    def validate_for_ci(self, max_unmapped_threshold: int = 50) -> bool:
        """
        Validate handler state for CI/CD pipelines.
        
        Args:
            max_unmapped_threshold: Maximum allowed unmapped tags
            
        Returns:
            True if validation passes, False otherwise
        """
        stats = self.get_usage_statistics()
        return stats['num_unmapped_tags'] <= max_unmapped_threshold

    def generate_safety_report(self, output_path: Optional[Path] = None) -> dict[str, Any]:
        """
        Generate a comprehensive safety report for flip decisions.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Dictionary containing the safety report
        """
        with self._stats_lock:
            stats = self.stats.copy()
        
        total_analyzed = stats.get('images_analyzed', 1)  # Avoid division by zero
        
        report = {
            'summary': {
                'total_images_analyzed': total_analyzed,
                'images_safely_flipped': stats.get('safe_flips', 0),
                'images_blocked_by_text': stats.get('blocked_by_text', 0),
                'images_blocked_by_unmapped': stats.get('blocked_by_safety', 0),
                'total_flips_performed': stats.get('total_flips', 0),
                'flip_rate': stats.get('safe_flips', 0) / total_analyzed,
                'block_rate': stats.get('skipped_flips', 0) / total_analyzed
            },
            'top_blocking_tags': sorted(
                stats.get('unmapped_blocking_frequency', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:50],
            'recommendations': self._generate_recommendations(stats)
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Safety report saved to {output_path}")
        
        return report
    
    def _generate_recommendations(self, stats: dict) -> list[str]:
        """Generate actionable recommendations based on statistics."""
        recommendations = []
        
        blocking_freq = stats.get('unmapped_blocking_frequency', {})
        if blocking_freq:
            top_blockers = sorted(blocking_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            total_blocked = sum(count for _, count in top_blockers)
            recommendations.append(
                f"Adding mappings for top 10 tags would unblock ~{total_blocked} images: "
                f"{', '.join(tag for tag, _ in top_blockers[:5])}"
            )
        
        if stats.get('blocked_by_text', 0) > stats.get('safe_flips', 0):
            recommendations.append(
                "Many images contain text/watermarks. Consider filtering these from training set."
            )
        
        return recommendations


class OrientationMonitor:
    """Monitor for tracking orientation handling health during training."""
    
    def __init__(self, threshold_unmapped: int = 10, check_interval: int = 100, out_dir: Optional[Path] = None):
        """
        Initialize the monitor.
        
        Args:
            threshold_unmapped: Number of unmapped tags before warning
            check_interval: How often to check (every N batches)
            out_dir: Optional directory to write diagnostics into (defaults to CWD)
        """
        self.threshold_unmapped = threshold_unmapped
        self.check_interval = check_interval
        self.warning_issued = False
        self.batch_count = 0
        self.last_unmapped_count = 0
        self.last_written_count = -1
        # Destination for diagnostics (default: CWD). Create it if missing.
        self.out_dir = Path(out_dir) if out_dir else Path.cwd()
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback so we never crash if an invalid path is provided.
            self.out_dir = Path.cwd()
        # Ensure the file exists from the start so it's produced by default.
        self._write_unmapped_file({'unmapped_tags_by_frequency': []}, handler=None)
    
    def check_health(self, handler: OrientationHandler, force: bool = False) -> None:
        """Check if too many unmapped tags are accumulating.
        
        Args:
            handler: The orientation handler to monitor
            force: Force check regardless of interval
        """
        self.batch_count += 1
        
        # Only check at intervals unless forced
        if not force and self.batch_count % self.check_interval != 0:
            return
        
        stats = handler.get_statistics()
        unmapped_count = stats['num_unmapped_tags']
        
        # Always (re)write the diagnostics file at each check when count changed; warning is separate.
        if unmapped_count != self.last_written_count:
            self._write_unmapped_file(stats, handler)
            self.last_written_count = unmapped_count

        # Check if unmapped tags are growing
        if unmapped_count > self.last_unmapped_count:
            new_unmapped = unmapped_count - self.last_unmapped_count
            if new_unmapped > 5:  # Significant growth
                logger.info(f"Found {new_unmapped} new unmapped orientation tags")
        
        # Issue warning if threshold exceeded
        if (unmapped_count > self.threshold_unmapped and not self.warning_issued):
            logger.warning(
                f"Found {unmapped_count} unmapped orientation tags. "
                f"Examples: {stats['unmapped_tags_sample'][:5]}. "
                "Consider updating orientation_map.json"
            )
            self.warning_issued = True
            # File was already refreshed above; nothing else to do here.
        
        self.last_unmapped_count = unmapped_count

    def _write_unmapped_file(self, stats: dict[str, Any], handler: Optional['OrientationHandler'] = None) -> None:
        """Write (or overwrite) unmapped_orientation_tags.txt atomically."""
        import os

        unmapped_file = self.out_dir / "unmapped_orientation_tags.txt"

        # Use process-specific temp file to avoid collisions with concurrent processes
        pid = os.getpid()
        tmp = self.out_dir / f"unmapped_orientation_tags.{pid}.tmp"

        try:
            suggestions = handler.suggest_mappings() if handler else {}
            with tmp.open('w', encoding='utf-8', newline='\n') as f:
                f.write("# Unmapped orientation tags found during training\n")
                f.write("# Format: tag (frequency) -> suggested_mapping\n\n")
                entries = stats.get('unmapped_tags_by_frequency') or []
                if not entries:
                    f.write("# No unmapped orientation tags encountered yet.\n")
                else:
                    for tag, freq in entries:
                        suggested = suggestions.get(tag, "# No suggestion")
                        f.write(f"{tag} ({freq}x) -> {suggested}\n")
            tmp.replace(unmapped_file)  # atomic on POSIX/NT
        except Exception as e:
            logger.error(f"Failed to write {unmapped_file}: {e}")
        finally:
            # Clean up temp file if it still exists
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass  # Ignore cleanup errors


def test_orientation_handler():
    """Test the orientation handler with various scenarios."""
    
    # Create test mapping file
    test_mapping = {
        "explicit_mappings": {
            "hair_over_left_eye": "hair_over_right_eye",
            "hair_over_right_eye": "hair_over_left_eye",
            "facing_left": "facing_right",
            "facing_right": "facing_left"
        },
        "regex_patterns": [
            {
                "pattern": "^left_(.*)",
                "replacement": "right_\\1",
                "description": "Tags starting with left_"
            },
            {
                "pattern": "^right_(.*)",
                "replacement": "left_\\1",
                "description": "Tags starting with right_"
            }
        ],
        "symmetric_tags": ["standing", "sitting", "looking_at_viewer"],
        "skip_flip_tags": ["text", "signature"]
    }
    
    # Save test mapping
    test_file = Path("test_orientation_map.json")
    with open(test_file, 'w') as f:
        json.dump(test_mapping, f)
    
    try:
        # Test with mapping file
        handler = OrientationHandler(
            mapping_file=test_file,
            random_flip_prob=0.5,
            strict_mode=True,
            safety_mode="balanced",
            skip_unmapped=True
        )
        
        # Test cases
        test_cases = [
            (["hair_over_left_eye", "standing"], ["hair_over_right_eye", "standing"]),
            (["facing_left", "text"], None),  # Should skip due to text
            (["left_hand", "right_foot"], ["right_hand", "left_foot"]),
            (["asymmetrical_hair", "standing"], ["asymmetrical_hair", "standing"]),  # Asymmetric tags don't block flips
            (["looking_at_viewer", "sitting"], ["looking_at_viewer", "sitting"]),  # Symmetric
            (["asymmetric_wings", "facing_left"], ["asymmetric_wings", "facing_right"])  # Asymmetric + orientation works
        ]
        
        print("Testing orientation handler...")
        for original_tags, expected in test_cases:
            result, flipped = handler.swap_tags(original_tags)
            
            if expected is None:
                assert not flipped, f"Should skip flip for {original_tags}"
                print(f"✓ Correctly skipped: {original_tags}")
            else:
                assert flipped, f"Should flip {original_tags}"
                assert result == expected, f"Expected {expected}, got {result}"
                print(f"✓ {original_tags} -> {result}")
        
        # Validate mappings
        issues = handler.validate_mappings()
        if issues:
            print(f"\nValidation issues found: {issues}")
        else:
            print("\n✓ All mappings valid")
        
        # Print statistics
        print(f"\nStatistics: {handler.get_statistics()}")

        # Test auto-suggestions
        test_unmapped = {"left_shoulder_tattoo", "right_knee_pad", "asymmetrical_outfit"}
        suggestions = handler.suggest_mappings(test_unmapped)
        print(f"\nAuto-suggestions: {suggestions}")
        
        # Test CI validation
        ci_valid = handler.validate_for_ci(max_unmapped_threshold=10)
        print(f"\nCI validation (threshold=10): {'PASS' if ci_valid else 'FAIL'}")
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    
    print("\nAll tests passed!")

def validate_dataset_orientation_tags(dataset_tags_file: Path, mapping_file: Path, 
                                      fail_on_unmapped: bool = False,
                                      max_unmapped_threshold: int = 50) -> int:
    """
    Standalone validation script for CI/CD pipelines.
    
    Args:
        dataset_tags_file: Path to file containing all dataset tags (one per line)
        mapping_file: Path to orientation mapping JSON
        fail_on_unmapped: If True, exit with error code if unmapped tags found
        max_unmapped_threshold: Maximum allowed unmapped tags before failure
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Load dataset tags
    with open(dataset_tags_file, 'r') as f:
        all_tags = set(line.strip() for line in f if line.strip())
    
    # Create handler and validate
    handler = OrientationHandler(mapping_file=mapping_file, strict_mode=True)
    issues = handler.validate_dataset_tags(all_tags)
    
    if issues:
        print("Validation issues found:")
        for issue_type, tags in issues.items():
            print(f"\n{issue_type}: {len(tags)} tags")
            for tag in tags[:10]:  # Show first 10
                print(f"  - {tag}")
    
    # Check threshold
    unmapped_count = len(issues.get('unmapped_orientation_tags', []))
    if fail_on_unmapped and unmapped_count > max_unmapped_threshold:
        print(f"\n❌ FAILED: {unmapped_count} unmapped tags exceeds threshold of {max_unmapped_threshold}")
        return 1
    
    print(f"\n✅ Validation passed: {unmapped_count} unmapped tags within threshold")
    return 0

if __name__ == "__main__":
    test_orientation_handler()

#!/usr/bin/env python3
"""
Enhanced Orientation Handler for Anime Image Tagger
Handles comprehensive left/right tag swapping for horizontal flip augmentation
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import warnings
import threading

logger = logging.getLogger(__name__)


class OrientationHandler:
    """
    Comprehensive handler for orientation-aware tag swapping during horizontal flips.
    Supports explicit mappings, regex patterns, and special handling rules.
    """
    
    def __init__(
        self,
        mapping_file: Optional[Path] = None,
        random_flip_prob: float = 0.0,
        strict_mode: bool = True,
        skip_unmapped: bool = False
    ):
        """
        Initialize the orientation handler.
        
        Args:
            mapping_file: Path to JSON file with orientation mappings
            random_flip_prob: Probability of random horizontal flips
            strict_mode: If True, fail if flips are enabled but no mapping provided
            skip_unmapped: If True, skip flipping images with unmapped orientation tags
        """
        self.random_flip_prob = random_flip_prob
        self.strict_mode = strict_mode
        self.skip_unmapped = skip_unmapped
        
        # Initialize mappings
        self.explicit_mappings = {}
        self.reverse_mappings = {}
        self.regex_patterns = []
        self.symmetric_tags = set()
        self.skip_flip_tags = set()
        self.complex_tags = {}
        
        # Track statistics (thread-safe)
        self.stats = {
            'total_flips': 0,
            'skipped_flips': 0,
            'unmapped_tags': set(),
            'mapped_tags': set()
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
    
    def _load_mappings(self, mapping_file: Path):
        """Load orientation mappings from JSON file."""
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load explicit mappings
            self.explicit_mappings = data.get('explicit_mappings', {})
            # Create reverse mappings for bidirectional swaps
            self.reverse_mappings = {v: k for k, v in self.explicit_mappings.items()}
            
            # Load regex patterns
            for pattern_data in data.get('regex_patterns', []):
                try:
                    pattern = re.compile(pattern_data['pattern'])
                    self.regex_patterns.append({
                        'pattern': pattern,
                        'replacement': pattern_data['replacement'],
                        'description': pattern_data.get('description', '')
                    })
                except re.error as e:
                    logger.error(f"Invalid regex pattern: {pattern_data['pattern']}: {e}")
            
            # Load symmetric tags (don't need swapping)
            self.symmetric_tags = set(data.get('symmetric_tags', []))
            
            # Load tags that should skip flipping
            self.skip_flip_tags = set(data.get('skip_flip_tags', []))
            
            # Load complex asymmetric tags
            self.complex_tags = data.get('complex_asymmetric_tags', {})
            
            logger.info(
                f"Loaded orientation mappings: "
                f"{len(self.explicit_mappings)} explicit, "
                f"{len(self.regex_patterns)} regex patterns, "
                f"{len(self.symmetric_tags)} symmetric, "
                f"{len(self.skip_flip_tags)} skip tags"
            )
            
        except Exception as e:
            if self.strict_mode:
                raise ValueError(f"Failed to load orientation mappings from {mapping_file}: {e}")
            else:
                logger.error(f"Failed to load orientation mappings: {e}. Using defaults.")
                self._load_default_mappings()
    
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
    
    def should_skip_flip(self, tags: List[str]) -> bool:
        """
        Check if image should skip flipping based on tags.
        
        Args:
            tags: List of tags for the image
            
        Returns:
            True if flip should be skipped
        """
        # Check for tags that should never be flipped (text, signatures, etc.)
        if any(tag in self.skip_flip_tags for tag in tags):
            with self._stats_lock:
                self.stats['skipped_flips'] += 1
            logger.debug(f"Skipping flip due to skip_flip_tag present")
            return True
        
        # Second check: if skip_unmapped is enabled, check for unmapped orientation tags
        if self.skip_unmapped:
            unmapped = self._find_unmapped_orientation_tags(tags)
            if unmapped:
                with self._stats_lock:
                    self.stats['skipped_flips'] += 1
                    self.stats['unmapped_tags'].update(unmapped)
                logger.debug(f"Skipping flip due to unmapped orientation tags: {unmapped}")
                return True
        
        return False
    
    def _find_unmapped_orientation_tags(self, tags: List[str]) -> Set[str]:
        """Find orientation tags that don't have mappings."""
        unmapped = set()
        orientation_keywords = ['left', 'right', 'asymmetric', 'single_', 'heterochromia']
        
        for tag in tags:
            # Skip if it's symmetric or already has explicit mapping
            if tag in self.symmetric_tags or tag in self.explicit_mappings:
                continue
            
            # Check if tag contains orientation keywords
            if any(keyword in tag for keyword in orientation_keywords):
                # Check if it matches any regex pattern
                matched = False
                for regex_data in self.regex_patterns:
                    if regex_data['pattern'].match(tag):
                        matched = True
                        break
                
                if not matched and tag not in self.explicit_mappings:
                    unmapped.add(tag)
        
        return unmapped
    
    def swap_tag(self, tag: str) -> str:
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
            with self._stats_lock:
                self.stats['mapped_tags'].add(tag)
            return self.explicit_mappings[tag]

        # Check reverse mappings
        if tag in self.reverse_mappings:
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
                        with self._stats_lock:
                            self.stats['mapped_tags'].add(tag)
                        return swapped
                except Exception as e:
                    logger.error(f"Regex substitution failed for tag '{tag}': {e}")

        # No mapping found - check if it's an orientation tag we should track
        if any(keyword in tag for keyword in ['left', 'right', 'asymmetric']):
            with self._stats_lock:
                self.stats['unmapped_tags'].add(tag)

        return tag
    
    def _swap_tag_unlocked(self, tag: str) -> str:
        """Internal version of swap_tag that doesn't acquire locks. For use when lock is already held."""
        return self.swap_tag(tag)   
    
    def swap_tags(self, tags: List[str]) -> Tuple[List[str], bool]:
        """
        Swap all tags for horizontal flip.

        Args:
            tags: List of original tags

        Returns:
            Tuple of (swapped tags, whether flip was applied)
        """
        # Check if we should skip this flip
        if self.should_skip_flip(tags):
            return tags, False

        # Swap all tags
        swapped_tags: List[str] = []
        for tag in tags:
            swapped_tags.append(self.swap_tag(tag))

        # Update flip counter
        with self._stats_lock:
            self.stats['total_flips'] += 1

        return swapped_tags, True
    
    def precompute_all_mappings(self, known_vocabulary: Set[str]) -> Dict[str, str]:
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
    
    def can_swap_eye_colors(self, eye_color_tags: List[str]) -> bool:
        """Check if eye color tags can be properly swapped for heterochromia."""
        # This is a simplified check - expand based on your tag format
        left_colors = [t for t in eye_color_tags if 'left' in t]
        right_colors = [t for t in eye_color_tags if 'right' in t]
        
        # Can only swap if we have both left and right eye color tags
        return len(left_colors) > 0 and len(right_colors) > 0
    
    def handle_complex_tags(self, tags: List[str]) -> Tuple[List[str], bool]:
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
    
    def validate_dataset_tags(self, all_tags: Set[str]) -> Dict[str, List[str]]:
        """Validate that all orientation-specific tags in dataset have mappings.
        
        Args:
            all_tags: Set of all unique tags in the dataset
            
        Returns:
            Dictionary of validation issues found
        """
        issues = {
            'unmapped_orientation_tags': [],
            'asymmetric_mappings': [],
            'conflicting_patterns': []
        }
        
        orientation_indicators = ['left', 'right', 'asymmetric', 'single_']
        
        for tag in all_tags:
            # Check if this might be an orientation tag
            if any(indicator in tag for indicator in orientation_indicators):
                # Skip if it's known to be symmetric
                if tag in self.symmetric_tags:
                    continue
                    
                # Check if this tag has a proper mapping
                swapped = self.swap_tag(tag)
                if swapped == tag:
                    issues['unmapped_orientation_tags'].append(tag)
                else:
                    # Verify bidirectionality
                    double_swapped = self.swap_tag(swapped)
                    if double_swapped != tag:
                        issues['asymmetric_mappings'].append(
                            f"{tag} -> {swapped} -> {double_swapped}"
                        )
        
        # Sort for consistent output
        for key in issues:
            issues[key] = sorted(issues[key])
        
        # Only return non-empty issues
        return {k: v for k, v in issues.items() if v}
    
    def validate_mappings(self) -> Dict[str, any]:
        """
        Validate the consistency of orientation mappings.
        
        Returns:
            Dictionary of validation issues found
        """
        issues = {
            'asymmetric_mappings': [],
            'circular_mappings': [],
            'duplicate_mappings': [],
            'regex_conflicts': []
        }
        
        # Check for asymmetric mappings (A->B but B doesn't map back to A)
        for tag1, tag2 in self.explicit_mappings.items():
            if tag2 in self.explicit_mappings:
                if self.explicit_mappings[tag2] != tag1:
                    issues['asymmetric_mappings'].append(f"{tag1} -> {tag2} -> {self.explicit_mappings[tag2]}")
        
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
    
    def get_usage_statistics(self) -> Dict[str, any]:
        """Get usage statistics."""
        with self._stats_lock:
            total_flips = self.stats['total_flips']
            skipped_flips = self.stats['skipped_flips']
            num_mapped = len(self.stats['mapped_tags'])
            num_unmapped = len(self.stats['unmapped_tags'])
            unmapped_sample = list(self.stats['unmapped_tags'])[:20]
        
        return {
            'total_flips': total_flips,
            'skipped_flips': skipped_flips,
            'skip_rate': skipped_flips / max(1, total_flips + skipped_flips),
            'num_mapped_tags': num_mapped,
            'num_unmapped_tags': num_unmapped,
            'unmapped_tags_sample': unmapped_sample
        }

    def get_statistics(self) -> Dict[str, any]:
        """Alias for get_usage_statistics for backwards compatibility."""
        return self.get_usage_statistics()


class OrientationMonitor:
    """Monitor for tracking orientation handling health during training."""
    
    def __init__(self, threshold_unmapped: int = 10, check_interval: int = 100):
        """
        Initialize the monitor.
        
        Args:
            threshold_unmapped: Number of unmapped tags before warning
            check_interval: How often to check (every N batches)
        """
        self.threshold_unmapped = threshold_unmapped
        self.check_interval = check_interval
        self.warning_issued = False
        self.batch_count = 0
        self.last_unmapped_count = 0
    
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
            
            # Optionally save unmapped tags to file for review
            unmapped_file = Path("unmapped_orientation_tags.txt")
            with open(unmapped_file, 'w') as f:
                f.write("# Unmapped orientation tags found during training\n")
                f.write("# Add these to orientation_map.json if needed\n\n")
                for tag in sorted(stats['unmapped_tags_sample']):
                    f.write(f"{tag}\n")
            logger.info(f"Saved unmapped tags to {unmapped_file}")
        
        self.last_unmapped_count = unmapped_count


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
            skip_unmapped=True
        )
        
        # Test cases
        test_cases = [
            (["hair_over_left_eye", "standing"], ["hair_over_right_eye", "standing"]),
            (["facing_left", "text"], None),  # Should skip due to text
            (["left_hand", "right_foot"], ["right_hand", "left_foot"]),
            (["asymmetrical_hair"], None),  # Should skip due to unmapped
            (["looking_at_viewer", "sitting"], ["looking_at_viewer", "sitting"])  # Symmetric
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
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_orientation_handler()
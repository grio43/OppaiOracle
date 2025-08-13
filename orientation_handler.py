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
        
        # Track statistics
        self.stats = {
            'total_flips': 0,
            'skipped_flips': 0,
            'unmapped_tags': set(),
            'mapped_tags': set()
        }
        
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
            self.stats['skipped_flips'] += 1
            return True
        
        # If skip_unmapped is enabled, check for unmapped orientation tags
        if self.skip_unmapped:
            unmapped = self._find_unmapped_orientation_tags(tags)
            if unmapped:
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
            self.stats['mapped_tags'].add(tag)
            return self.explicit_mappings[tag]
        
        # Check reverse mappings
        if tag in self.reverse_mappings:
            self.stats['mapped_tags'].add(tag)
            return self.reverse_mappings[tag]
        
        # Try regex patterns
        for regex_data in self.regex_patterns:
            pattern = regex_data['pattern']
            if pattern.match(tag):
                try:
                    swapped = pattern.sub(regex_data['replacement'], tag)
                    if swapped != tag:  # Only count if actually changed
                        self.stats['mapped_tags'].add(tag)
                        return swapped
                except Exception as e:
                    logger.error(f"Regex substitution failed for tag '{tag}': {e}")
        
        # No mapping found
        if any(keyword in tag for keyword in ['left', 'right', 'asymmetric']):
            self.stats['unmapped_tags'].add(tag)
            if self.strict_mode:
                logger.warning(f"No orientation mapping for tag: {tag}")
        
        return tag
    
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
        swapped_tags = [self.swap_tag(tag) for tag in tags]
        self.stats['total_flips'] += 1
        
        return swapped_tags, True
    
    def validate_mappings(self) -> Dict[str, List[str]]:
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
    
    def get_statistics(self) -> Dict[str, any]:
        """Get usage statistics."""
        return {
            'total_flips': self.stats['total_flips'],
            'skipped_flips': self.stats['skipped_flips'],
            'skip_rate': self.stats['skipped_flips'] / max(1, self.stats['total_flips'] + self.stats['skipped_flips']),
            'num_mapped_tags': len(self.stats['mapped_tags']),
            'num_unmapped_tags': len(self.stats['unmapped_tags']),
            'unmapped_tags_sample': list(self.stats['unmapped_tags'])[:20]  # First 20 unmapped tags
        }


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
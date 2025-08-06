#!/usr/bin/env python3
"""
Tag Vocabulary Manager for Anime Image Tagger
Manages 200k tag vocabulary with hierarchical grouping and special handling
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
import pickle
import re
from enum import Enum
import yaml
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TagType(Enum):
    """Danbooru tag categories"""
    GENERAL = 0      # General descriptive tags
    ARTIST = 1       # artist:* tags
    CHARACTER = 3    # character:* tags  
    COPYRIGHT = 4    # copyright:* tags (series/franchise)
    META = 5         # meta:* tags (highres, etc)
    
    @classmethod
    def from_tag(cls, tag: str) -> 'TagType':
        """Determine tag type from tag string"""
        if tag.startswith('artist:'):
            return cls.ARTIST
        elif tag.startswith('character:'):
            return cls.CHARACTER
        elif tag.startswith('copyright:'):
            return cls.COPYRIGHT
        elif tag.startswith('meta:'):
            return cls.META
        else:
            return cls.GENERAL


@dataclass
class TagInfo:
    """Information about a single tag"""
    tag: str
    index: int
    count: int = 0
    type: TagType = TagType.GENERAL
    aliases: Set[str] = field(default_factory=set)
    implications: Set[str] = field(default_factory=set)  # Tags this implies
    group_id: int = -1  # Which of the 20 groups
    group_index: int = -1  # Index within that group


class TagVocabulary:
    """Manages tag vocabulary for anime image tagger"""
    
    def __init__(self, 
                 vocab_file: Optional[Path] = None,
                 num_groups: int = 20,
                 tags_per_group: int = 10000,
                 total_tags: int = 200000):
        
        self.num_groups = num_groups
        self.tags_per_group = tags_per_group
        self.total_tags = total_tags
        
        # Core mappings
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_info: Dict[str, TagInfo] = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_index = 0
        self.unk_index = 1
        
        # Tag statistics
        self.tag_counts: Counter = Counter()
        self.tag_cooccurrence: Dict[str, Counter] = defaultdict(Counter)  # Fixed type annotation
        
        # Hierarchical grouping
        self.groups: List[Set[str]] = [set() for _ in range(num_groups)]
        self.group_assignments: Dict[str, int] = {}
        
        # Special tag categories
        self.special_tags = {
            'ratings': {'safe', 'questionable', 'explicit', 'general', 'sensitive'},
            'quality': {'best_quality', 'high_quality', 'normal_quality', 'low_quality', 'worst_quality'},
            'status': {'translation_request', 'check_translation', 'translated', 'partially_translated'},
            'meta': {'highres', 'absurdres', 'incredibly_absurdres', 'huge_filesize', 'wallpaper'}
        }
        
        # Initialize vocabulary
        if vocab_file and Path(vocab_file).exists():
            self.load_vocabulary(vocab_file)
        else:
            self._initialize_empty_vocabulary()
    
    def _initialize_empty_vocabulary(self):
        """Initialize with special tokens only"""
        self.tag_to_index[self.pad_token] = self.pad_index
        self.tag_to_index[self.unk_token] = self.unk_index
        self.index_to_tag[self.pad_index] = self.pad_token
        self.index_to_tag[self.unk_index] = self.unk_token
        
        self.tag_info[self.pad_token] = TagInfo(self.pad_token, self.pad_index)
        self.tag_info[self.unk_token] = TagInfo(self.unk_token, self.unk_index)
        
        logger.info("Initialized empty vocabulary with special tokens")
    
    def load_vocabulary(self, vocab_file: Path):
        """Load vocabulary from file"""
        vocab_file = Path(vocab_file)
        
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        
        if vocab_file.suffix == '.json':
            self._load_json_vocabulary(vocab_file)
        elif vocab_file.suffix == '.txt':
            self._load_text_vocabulary(vocab_file)
        elif vocab_file.suffix == '.pkl':
            self._load_pickle_vocabulary(vocab_file)
        else:
            raise ValueError(f"Unknown vocabulary format: {vocab_file.suffix}")
            
        logger.info(f"Loaded {len(self.tag_to_index)} tags from {vocab_file}")
    
    def _load_json_vocabulary(self, vocab_file: Path):
        """Load vocabulary from JSON format"""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in vocabulary file: {e}")
        
        if isinstance(data, list):
            # Simple list of tags
            self._initialize_empty_vocabulary()
            for idx, tag in enumerate(data, start=2):  # Start after special tokens
                if isinstance(tag, str):  # Validate tag is string
                    self.add_tag(tag, index=idx)
        else:
            # Full vocabulary dump
            self.tag_to_index = data.get('tag_to_index', {})
            self.index_to_tag = {int(k): v for k, v in data.get('index_to_tag', {}).items()}
            
            # Reconstruct tag info with proper error handling
            for tag, idx in self.tag_to_index.items():
                tag_type_value = data.get('tag_types', {}).get(tag, 0)
                try:
                    tag_type = TagType(tag_type_value)
                except ValueError:
                    tag_type = TagType.GENERAL
                    
                self.tag_info[tag] = TagInfo(
                    tag=tag,
                    index=idx,
                    count=data.get('tag_counts', {}).get(tag, 0),
                    type=tag_type
                )
            
            # Load group assignments if present
            if 'group_assignments' in data:
                self.group_assignments = data['group_assignments']
                # Reconstruct groups
                for tag, group_id in self.group_assignments.items():
                    if 0 <= group_id < self.num_groups:
                        self.groups[group_id].add(tag)
    
    def _load_text_vocabulary(self, vocab_file: Path):
        """Load vocabulary from text file (one tag per line)"""
        self._initialize_empty_vocabulary()
        
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f, start=2):  # Start after special tokens
                    tag = line.strip()
                    if tag:
                        # Parse count if present (format: "tag\tcount")
                        if '\t' in tag:
                            parts = tag.split('\t', 1)
                            if len(parts) == 2:
                                tag = parts[0]
                                try:
                                    count = int(parts[1])
                                except ValueError:
                                    count = 0
                        else:
                            count = 0
                        
                        self.add_tag(tag, index=idx, count=count)
        except IOError as e:
            raise IOError(f"Error reading vocabulary file: {e}")
    
    def _load_pickle_vocabulary(self, vocab_file: Path):
        """Load vocabulary from pickle file"""
        try:
            with open(vocab_file, 'rb') as f:
                data = pickle.load(f)
                # Validate data structure before updating
                if isinstance(data, dict):
                    self.__dict__.update(data)
                else:
                    raise ValueError("Invalid pickle data structure")
        except (pickle.PickleError, EOFError) as e:
            raise ValueError(f"Error loading pickle file: {e}")
    
    def save_vocabulary(self, output_path: Path, format: str = 'json'):
        """Save vocabulary to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._save_json_vocabulary(output_path)
        elif format == 'txt':
            self._save_text_vocabulary(output_path)
        elif format == 'pkl':
            self._save_pickle_vocabulary(output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        logger.info(f"Saved vocabulary to {output_path}")
    
    def _save_json_vocabulary(self, output_path: Path):
        """Save vocabulary in JSON format"""
        data = {
            'version': '1.0',
            'num_tags': len(self.tag_to_index),
            'num_groups': self.num_groups,
            'tags_per_group': self.tags_per_group,
            'tag_to_index': self.tag_to_index,
            'index_to_tag': {str(k): v for k, v in self.index_to_tag.items()},
            'tag_counts': dict(self.tag_counts),
            'tag_types': {tag: info.type.value for tag, info in self.tag_info.items()},
            'group_assignments': self.group_assignments,
            'special_tags': self.special_tags
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Error saving vocabulary: {e}")
    
    def _save_text_vocabulary(self, output_path: Path):
        """Save vocabulary as text file with counts"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Sort by frequency
                sorted_tags = sorted(self.tag_info.values(), 
                                   key=lambda x: x.count, 
                                   reverse=True)
                
                for tag_info in sorted_tags:
                    if tag_info.tag not in [self.pad_token, self.unk_token]:
                        f.write(f"{tag_info.tag}\t{tag_info.count}\n")
        except IOError as e:
            raise IOError(f"Error saving text vocabulary: {e}")
    
    def _save_pickle_vocabulary(self, output_path: Path):
        """Save vocabulary as pickle"""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        except IOError as e:
            raise IOError(f"Error saving pickle vocabulary: {e}")
    
    def add_tag(self, tag: str, index: Optional[int] = None, count: int = 0) -> int:
        """Add a tag to vocabulary"""
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"Invalid tag: {tag}")
            
        if tag in self.tag_to_index:
            # Update count if tag exists
            self.tag_info[tag].count += count
            self.tag_counts[tag] += count
            return self.tag_to_index[tag]
        
        # Assign index
        if index is None:
            index = len(self.tag_to_index)
        
        # Check if we've reached the maximum number of tags
        if len(self.tag_to_index) >= self.total_tags:
            logger.warning(f"Maximum number of tags ({self.total_tags}) reached")
            return self.unk_index
        
        # Add to mappings
        self.tag_to_index[tag] = index
        self.index_to_tag[index] = tag
        
        # Create tag info
        tag_type = TagType.from_tag(tag)
        self.tag_info[tag] = TagInfo(
            tag=tag,
            index=index,
            count=count,
            type=tag_type
        )
        
        # Update counts
        self.tag_counts[tag] = count
        
        return index
    
    def build_from_dataset(self, 
                          tag_files: List[Path],
                          min_count: int = 10,
                          max_tags: int = 200000):
        """Build vocabulary from dataset tag files"""
        if not tag_files:
            raise ValueError("No tag files provided")
            
        logger.info(f"Building vocabulary from {len(tag_files)} files")
        
        # Count all tags
        all_tags = Counter()
        tag_cooccurrence = defaultdict(Counter)
        
        for tag_file in tag_files:
            try:
                tag_file = Path(tag_file)
                if not tag_file.exists():
                    logger.warning(f"Tag file not found: {tag_file}")
                    continue
                    
                with open(tag_file, 'r', encoding='utf-8') as f:
                    tags = json.load(f)
                    
                if isinstance(tags, list):
                    # Filter out empty strings and validate
                    tags = [t for t in tags if isinstance(t, str) and t.strip()]
                    all_tags.update(tags)
                    
                    # Track co-occurrence
                    for i, tag1 in enumerate(tags):
                        for tag2 in tags[i+1:]:
                            tag_cooccurrence[tag1][tag2] += 1
                            tag_cooccurrence[tag2][tag1] += 1
                            
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading {tag_file}: {e}")
        
        if not all_tags:
            raise ValueError("No valid tags found in provided files")
        
        # Filter by minimum count
        filtered_tags = [(tag, count) for tag, count in all_tags.items() 
                        if count >= min_count]
        
        if not filtered_tags:
            logger.warning(f"No tags meet minimum count threshold of {min_count}")
            filtered_tags = list(all_tags.items())[:max_tags]
        
        # Sort by frequency
        filtered_tags.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max tags
        if len(filtered_tags) > max_tags:
            filtered_tags = filtered_tags[:max_tags]
        
        # Initialize vocabulary
        self._initialize_empty_vocabulary()
        
        # Add tags in frequency order
        for tag, count in filtered_tags:
            self.add_tag(tag, count=count)
        
        # Store co-occurrence data
        self.tag_cooccurrence = dict(tag_cooccurrence)  # Convert from defaultdict
        
        logger.info(f"Built vocabulary with {len(self.tag_to_index)} tags")
        
        # Assign to hierarchical groups
        self.assign_hierarchical_groups()
    
    def assign_hierarchical_groups(self, strategy: str = 'frequency_balanced'):
        """Assign tags to hierarchical groups for the model head"""
        logger.info(f"Assigning tags to {self.num_groups} groups using {strategy} strategy")
        
        # Clear existing assignments
        self.groups = [set() for _ in range(self.num_groups)]
        self.group_assignments.clear()
        
        # Get all tags except special tokens
        all_tags = [tag for tag in self.tag_to_index 
                   if tag not in [self.pad_token, self.unk_token]]
        
        if not all_tags:
            logger.warning("No tags to assign to groups")
            return
        
        if strategy == 'frequency_balanced':
            self._assign_frequency_balanced_groups(all_tags)
        elif strategy == 'type_based':
            self._assign_type_based_groups(all_tags)
        elif strategy == 'cooccurrence':
            self._assign_cooccurrence_groups(all_tags)
        else:
            raise ValueError(f"Unknown grouping strategy: {strategy}")
        
        # Update tag info with group assignments
        for group_id, group_tags in enumerate(self.groups):
            for group_idx, tag in enumerate(sorted(group_tags)):
                if tag in self.tag_info:
                    self.tag_info[tag].group_id = group_id
                    self.tag_info[tag].group_index = group_idx
        
        logger.info("Group assignment complete")
        self._log_group_statistics()
    
    def _assign_frequency_balanced_groups(self, tags: List[str]):
        """Assign tags to groups balancing frequency across groups"""
        # Sort by frequency
        sorted_tags = sorted(tags, key=lambda t: self.tag_counts.get(t, 0), reverse=True)
        
        # Track group sizes to prevent overflow
        group_sizes = [0] * self.num_groups
        
        # Round-robin assignment to balance frequencies
        for idx, tag in enumerate(sorted_tags):
            # Find the group with least tags that isn't full
            min_group = min(range(self.num_groups), 
                          key=lambda g: (group_sizes[g], g))
            
            if group_sizes[min_group] < self.tags_per_group:
                self.groups[min_group].add(tag)
                self.group_assignments[tag] = min_group
                group_sizes[min_group] += 1
            else:
                # All groups are full, log warning
                logger.warning(f"All groups full, cannot assign tag: {tag}")
                break
    
    def _assign_type_based_groups(self, tags: List[str]):
        """Assign tags to groups based on tag type"""
        # Separate by type
        tags_by_type = defaultdict(list)
        for tag in tags:
            if tag in self.tag_info:
                tag_type = self.tag_info[tag].type
                tags_by_type[tag_type].append(tag)
        
        # Reserve groups for special types
        group_id = 0
        
        # Artist tags get their own groups
        if TagType.ARTIST in tags_by_type:
            artist_tags = sorted(tags_by_type[TagType.ARTIST], 
                               key=lambda t: self.tag_counts.get(t, 0), reverse=True)
            artist_groups = min(3, self.num_groups // 4)  # Up to 3 groups for artists
            
            for i in range(artist_groups):
                if group_id >= self.num_groups:
                    break
                    
                start_idx = i * len(artist_tags) // artist_groups
                end_idx = (i + 1) * len(artist_tags) // artist_groups
                
                for tag in artist_tags[start_idx:end_idx]:
                    if len(self.groups[group_id]) < self.tags_per_group:
                        self.groups[group_id].add(tag)
                        self.group_assignments[tag] = group_id
                
                group_id += 1
        
        # Character tags
        if TagType.CHARACTER in tags_by_type:
            char_tags = sorted(tags_by_type[TagType.CHARACTER],
                             key=lambda t: self.tag_counts.get(t, 0), reverse=True)
            char_groups = min(3, self.num_groups // 4)
            
            for i in range(char_groups):
                if group_id >= self.num_groups:
                    break
                    
                start_idx = i * len(char_tags) // char_groups
                end_idx = (i + 1) * len(char_tags) // char_groups
                
                for tag in char_tags[start_idx:end_idx]:
                    if len(self.groups[group_id]) < self.tags_per_group:
                        self.groups[group_id].add(tag)
                        self.group_assignments[tag] = group_id
                
                group_id += 1
        
        # Remaining groups for general tags
        general_tags = sorted(tags_by_type[TagType.GENERAL],
                            key=lambda t: self.tag_counts.get(t, 0), reverse=True)
        
        for tag in general_tags:
            # Find group with space
            assigned = False
            for gid in range(group_id, self.num_groups):
                if len(self.groups[gid]) < self.tags_per_group:
                    self.groups[gid].add(tag)
                    self.group_assignments[tag] = gid
                    assigned = True
                    break
            
            if not assigned:
                # Start filling from beginning if needed
                for gid in range(self.num_groups):
                    if len(self.groups[gid]) < self.tags_per_group:
                        self.groups[gid].add(tag)
                        self.group_assignments[tag] = gid
                        break
    
    def _assign_cooccurrence_groups(self, tags: List[str]):
        """Assign tags to groups based on co-occurrence patterns"""
        assigned = set()
        group_id = 0
        
        # Start with most frequent tags as seeds
        sorted_tags = sorted(tags, key=lambda t: self.tag_counts.get(t, 0), reverse=True)
        
        for seed_tag in sorted_tags:
            if seed_tag in assigned:
                continue
                
            if group_id >= self.num_groups:
                break
                
            # Add seed to group
            self.groups[group_id].add(seed_tag)
            self.group_assignments[seed_tag] = group_id
            assigned.add(seed_tag)
            
            # Add frequently co-occurring tags
            if seed_tag in self.tag_cooccurrence:
                cooccurring = self.tag_cooccurrence[seed_tag].most_common(self.tags_per_group - 1)
                
                for tag, _ in cooccurring:
                    if tag not in assigned and len(self.groups[group_id]) < self.tags_per_group:
                        self.groups[group_id].add(tag)
                        self.group_assignments[tag] = group_id
                        assigned.add(tag)
            
            group_id += 1
        
        # Assign remaining tags
        for tag in tags:
            if tag not in assigned:
                # Find group with space
                for gid in range(self.num_groups):
                    if len(self.groups[gid]) < self.tags_per_group:
                        self.groups[gid].add(tag)
                        self.group_assignments[tag] = gid
                        assigned.add(tag)
                        break
    
    def _log_group_statistics(self):
        """Log statistics about group assignments"""
        for i, group in enumerate(self.groups):
            if group:
                total_count = sum(self.tag_counts.get(tag, 0) for tag in group)
                avg_count = total_count / len(group) if group else 0
                
                # Find most common tags in group
                top_tags = sorted(group, 
                                key=lambda t: self.tag_counts.get(t, 0), 
                                reverse=True)[:5]
                
                logger.info(f"Group {i}: {len(group)} tags, "
                          f"avg count: {avg_count:.1f}, "
                          f"top tags: {', '.join(top_tags)}")
    
    def get_tag_index(self, tag: str) -> int:
        """Get index for tag, return UNK if not found"""
        return self.tag_to_index.get(tag, self.unk_index)
    
    def get_tag_indices(self, tags: List[str]) -> List[int]:
        """Convert list of tags to indices"""
        return [self.get_tag_index(tag) for tag in tags]
    
    def get_tag_from_index(self, index: int) -> str:
        """Get tag from index"""
        return self.index_to_tag.get(index, self.unk_token)
    
    def get_tags_from_indices(self, indices: List[int]) -> List[str]:
        """Convert list of indices to tags"""
        return [self.get_tag_from_index(idx) for idx in indices]
    
    def get_group_indices(self, tag: str) -> Tuple[int, int]:
        """Get hierarchical group indices for tag"""
        if tag in self.tag_info:
            info = self.tag_info[tag]
            return info.group_id, info.group_index
        return -1, -1
    
    def encode_tags_hierarchical(self, tags: List[str]) -> Dict[int, List[int]]:
        """Encode tags into hierarchical group format"""
        group_tags: Dict[int, List[int]] = defaultdict(list)
        
        for tag in tags:
            group_id, group_idx = self.get_group_indices(tag)
            if group_id >= 0 and group_idx >= 0:
                group_tags[group_id].append(group_idx)
        
        return dict(group_tags)
    
    def add_tag_implications(self, implications_file: Path):
        """Load tag implications (e.g., cat_ears -> animal_ears)"""
        implications_file = Path(implications_file)
        if not implications_file.exists():
            logger.warning(f"Implications file not found: {implications_file}")
            return
            
        try:
            with open(implications_file, 'r', encoding='utf-8') as f:
                implications = json.load(f)
            
            for tag, implied_tags in implications.items():
                if tag in self.tag_info and isinstance(implied_tags, list):
                    self.tag_info[tag].implications.update(implied_tags)
                    
            logger.info(f"Loaded implications from {implications_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading implications: {e}")
    
    def add_tag_aliases(self, aliases_file: Path):
        """Load tag aliases"""
        aliases_file = Path(aliases_file)
        if not aliases_file.exists():
            logger.warning(f"Aliases file not found: {aliases_file}")
            return
            
        try:
            with open(aliases_file, 'r', encoding='utf-8') as f:
                aliases = json.load(f)
            
            for main_tag, alias_list in aliases.items():
                if main_tag in self.tag_info and isinstance(alias_list, list):
                    self.tag_info[main_tag].aliases.update(alias_list)
                    
                    # Map aliases to main tag
                    for alias in alias_list:
                        if isinstance(alias, str) and alias:
                            self.tag_to_index[alias] = self.tag_to_index[main_tag]
                            
            logger.info(f"Loaded aliases from {aliases_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading aliases: {e}")
    
    def get_special_tag_indices(self, category: str) -> List[int]:
        """Get indices for special tag categories"""
        if category in self.special_tags:
            tags = self.special_tags[category]
            return [self.get_tag_index(tag) for tag in tags if tag in self.tag_to_index]
        return []
    
    def compute_tag_statistics(self) -> Dict:
        """Compute various statistics about the vocabulary"""
        stats = {
            'total_tags': len(self.tag_to_index),
            'total_occurrences': sum(self.tag_counts.values()),
            'tags_by_type': defaultdict(int),
            'tags_by_frequency': defaultdict(int),
            'group_sizes': [len(g) for g in self.groups],
            'group_total_counts': []
        }
        
        # Count by type
        for tag_info in self.tag_info.values():
            stats['tags_by_type'][tag_info.type.name] += 1
        
        # Frequency buckets
        buckets = [(1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, float('inf'))]
        for tag, count in self.tag_counts.items():
            for min_c, max_c in buckets:
                if min_c <= count < max_c:
                    bucket_name = f'{min_c}-{max_c}' if max_c != float('inf') else f'{min_c}+'
                    stats['tags_by_frequency'][bucket_name] += 1
                    break
        
        # Group statistics
        for group in self.groups:
            if group:
                total = sum(self.tag_counts.get(tag, 0) for tag in group)
                stats['group_total_counts'].append(total)
            else:
                stats['group_total_counts'].append(0)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['tags_by_type'] = dict(stats['tags_by_type'])
        stats['tags_by_frequency'] = dict(stats['tags_by_frequency'])
        
        return stats
    
    def export_for_training(self, output_dir: Path):
        """Export vocabulary in format needed for training"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main vocabulary
        self.save_vocabulary(output_dir / 'vocabulary.json', format='json')
        
        # Save group mappings for hierarchical head
        group_data = {
            'num_groups': self.num_groups,
            'tags_per_group': self.tags_per_group,
            'group_assignments': {}
        }
        
        for group_id, group_tags in enumerate(self.groups):
            if group_tags:  # Only save non-empty groups
                group_data['group_assignments'][str(group_id)] = {
                    tag: self.tag_info[tag].group_index 
                    for tag in sorted(group_tags)
                    if tag in self.tag_info
                }
        
        try:
            with open(output_dir / 'group_mappings.json', 'w', encoding='utf-8') as f:
                json.dump(group_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving group mappings: {e}")
        
        # Save statistics
        stats = self.compute_tag_statistics()
        try:
            with open(output_dir / 'vocabulary_stats.json', 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving statistics: {e}")
        
        logger.info(f"Exported vocabulary to {output_dir}")


# Utility functions for common operations
def create_vocabulary_from_danbooru(
    tag_files: List[Path],
    output_dir: Path,
    min_count: int = 10,
    max_tags: int = 200000
) -> TagVocabulary:
    """Create vocabulary from Danbooru-style tag files"""
    
    vocab = TagVocabulary(
        num_groups=20,
        tags_per_group=10000,
        total_tags=max_tags
    )
    
    # Build from dataset
    vocab.build_from_dataset(tag_files, min_count=min_count, max_tags=max_tags)
    
    # Export for training
    vocab.export_for_training(output_dir)
    
    return vocab


def load_vocabulary_for_training(vocab_dir: Path) -> TagVocabulary:
    """Load vocabulary for model training"""
    vocab_dir = Path(vocab_dir)
    vocab_file = vocab_dir / 'vocabulary.json'
    
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    
    vocab = TagVocabulary(vocab_file=vocab_file)
    
    # Load group mappings if available
    group_file = vocab_dir / 'group_mappings.json'
    if group_file.exists():
        try:
            with open(group_file, 'r', encoding='utf-8') as f:
                group_data = json.load(f)
                
            # Reconstruct group assignments
            for group_id, assignments in group_data.get('group_assignments', {}).items():
                for tag, group_idx in assignments.items():
                    if tag in vocab.tag_info:
                        vocab.tag_info[tag].group_id = int(group_id)
                        vocab.tag_info[tag].group_index = group_idx
                        
            logger.info(f"Loaded group mappings from {group_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading group mappings: {e}")
    
    return vocab


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build tag vocabulary for anime image tagger")
    parser.add_argument('--tag_files', type=str, nargs='+', 
                       help='Paths to tag JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for vocabulary')
    parser.add_argument('--min_count', type=int, default=10,
                       help='Minimum tag frequency (default: 10)')
    parser.add_argument('--max_tags', type=int, default=200000,
                       help='Maximum number of tags (default: 200000)')
    parser.add_argument('--grouping_strategy', type=str, default='frequency_balanced',
                       choices=['frequency_balanced', 'type_based', 'cooccurrence'],
                       help='Strategy for hierarchical grouping (default: frequency_balanced)')
    
    args = parser.parse_args()
    
    try:
        if args.tag_files:
            tag_files = [Path(f) for f in args.tag_files]
            
            # Validate that files exist
            missing_files = [f for f in tag_files if not f.exists()]
            if missing_files:
                logger.error(f"Missing tag files: {missing_files}")
                sys.exit(1)
            
            vocab = create_vocabulary_from_danbooru(
                tag_files=tag_files,
                output_dir=Path(args.output_dir),
                min_count=args.min_count,
                max_tags=args.max_tags
            )
            
            # Apply specified grouping strategy
            vocab.assign_hierarchical_groups(strategy=args.grouping_strategy)
            
            # Re-export with new grouping
            vocab.export_for_training(Path(args.output_dir))
            
            # Print statistics
            stats = vocab.compute_tag_statistics()
            print(f"\nVocabulary Statistics:")
            print(f"Total tags: {stats['total_tags']:,}")
            print(f"Total occurrences: {stats['total_occurrences']:,}")
            print(f"\nTags by type:")
            for tag_type, count in sorted(stats['tags_by_type'].items()):
                print(f"  {tag_type}: {count:,}")
            print(f"\nTags by frequency:")
            for bucket, count in sorted(stats['tags_by_frequency'].items()):
                print(f"  {bucket}: {count:,}")
            print(f"\nGroup sizes:")
            for i, size in enumerate(stats['group_sizes']):
                if size > 0:
                    print(f"  Group {i}: {size:,} tags")
        else:
            logger.error("No tag files specified. Use --tag_files option.")
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
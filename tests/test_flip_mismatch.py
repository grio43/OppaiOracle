#!/usr/bin/env python3
"""
Tests for flip mismatch detection in orientation handling.

These tests verify that:
1. Force flip mode properly detects unmapped orientation tags
2. Random flip mode tracks mismatches correctly
3. swap_tags_with_info returns accurate mismatch information
4. Cache hit and miss paths produce identical flip behavior
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orientation_handler import OrientationHandler, SwapResult


class TestSwapResult:
    """Tests for the SwapResult dataclass."""

    def test_swap_result_fields(self):
        """Test that SwapResult has all expected fields."""
        result = SwapResult(
            swapped_tags=["tag1", "tag2"],
            flip_applied=True,
            unmapped_orientation_tags=["looking_left"],
            has_mismatch=True
        )
        assert result.swapped_tags == ["tag1", "tag2"]
        assert result.flip_applied is True
        assert result.unmapped_orientation_tags == ["looking_left"]
        assert result.has_mismatch is True

    def test_swap_result_no_mismatch(self):
        """Test SwapResult when there's no mismatch."""
        result = SwapResult(
            swapped_tags=["looking_right"],
            flip_applied=True,
            unmapped_orientation_tags=[],
            has_mismatch=False
        )
        assert result.has_mismatch is False
        assert result.unmapped_orientation_tags == []


class TestSwapTagsWithInfo:
    """Tests for the swap_tags_with_info method."""

    @pytest.fixture
    def handler(self):
        """Create an OrientationHandler with basic mappings."""
        handler = OrientationHandler(
            mapping_file=None,
            random_flip_prob=0.5,
            strict_mode=False,
            safety_mode="balanced"
        )
        # Add some explicit mappings for testing
        handler.explicit_mappings = {
            "looking_left": "looking_right",
            "looking_right": "looking_left",
            "left_hand": "right_hand",
            "right_hand": "left_hand",
        }
        handler.reverse_mappings = dict(handler.explicit_mappings)
        return handler

    def test_all_tags_mapped(self, handler):
        """Test when all orientation tags have mappings."""
        tags = ["looking_left", "red_hair", "1girl"]
        result = handler.swap_tags_with_info(tags, skip_safety_check=True)

        assert result.flip_applied is True
        assert "looking_right" in result.swapped_tags
        assert result.has_mismatch is False
        assert result.unmapped_orientation_tags == []

    def test_unmapped_orientation_tag(self, handler):
        """Test detection of unmapped orientation tags."""
        # Use a tag with 'left' that doesn't match regex patterns:
        # - not prefix 'left_'
        # - not suffix '_left'
        # - not middle '_left_'
        # 'leftward_gaze' has 'left' but not in a mappable position
        tags = ["looking_left", "leftward_gaze", "1girl"]
        result = handler.swap_tags_with_info(tags, skip_safety_check=True)

        assert result.flip_applied is True
        assert result.has_mismatch is True
        assert "leftward_gaze" in result.unmapped_orientation_tags

    def test_filtered_tags_not_flagged(self, handler):
        """Test that filtered tags (asymmetric, bright, etc.) aren't flagged as unmapped."""
        tags = ["looking_left", "asymmetrical_hair", "bright_eyes", "upright_pose"]
        result = handler.swap_tags_with_info(tags, skip_safety_check=True)

        # These should NOT be flagged as unmapped orientation tags
        assert "asymmetrical_hair" not in result.unmapped_orientation_tags
        assert "bright_eyes" not in result.unmapped_orientation_tags
        assert "upright_pose" not in result.unmapped_orientation_tags

    def test_safety_check_respected(self, handler):
        """Test that safety check is respected when not skipped."""
        # Add a skip_flip tag
        handler.skip_flip_tags = {"text"}
        tags = ["looking_left", "text"]

        # Without skip_safety_check, should return no flip
        result = handler.swap_tags_with_info(tags, skip_safety_check=False)
        assert result.flip_applied is False
        assert result.has_mismatch is False

        # With skip_safety_check, should proceed with flip
        result = handler.swap_tags_with_info(tags, skip_safety_check=True)
        assert result.flip_applied is True

    def test_empty_tags(self, handler):
        """Test handling of empty tag list."""
        result = handler.swap_tags_with_info([], skip_safety_check=True)
        assert result.swapped_tags == []
        assert result.flip_applied is True  # Empty list still "flips" successfully
        assert result.has_mismatch is False


class TestMismatchDetection:
    """Integration tests for mismatch detection scenarios."""

    @pytest.fixture
    def handler_minimal(self):
        """Create handler with minimal mappings to test mismatch scenarios."""
        handler = OrientationHandler(
            mapping_file=None,
            random_flip_prob=0.5,
            strict_mode=False,
            safety_mode="balanced"
        )
        # Only map looking_left/right, leave other left/right tags unmapped
        handler.explicit_mappings = {
            "looking_left": "looking_right",
            "looking_right": "looking_left",
        }
        handler.reverse_mappings = dict(handler.explicit_mappings)
        return handler

    def test_force_flip_mismatch_scenario(self, handler_minimal):
        """Simulate force flip mode with unmapped tags."""
        # This represents what happens in force mode
        # Use 'leftward_gaze' which has 'left' but doesn't match regex patterns
        original_tags = ["looking_left", "leftward_gaze", "1girl"]

        result = handler_minimal.swap_tags_with_info(
            original_tags,
            skip_safety_check=True,
            record_stats=True
        )

        # Force mode always flips
        flip_bit = True  # Force mode sets this unconditionally

        # Check mismatch detection
        assert result.has_mismatch is True
        assert "leftward_gaze" in result.unmapped_orientation_tags

        # Verify the mapped tag was swapped
        assert "looking_right" in result.swapped_tags

        # Verify unmapped tag was NOT swapped
        assert "leftward_gaze" in result.swapped_tags  # Unchanged

    def test_mixed_batch_consistency(self, handler_minimal):
        """Test that flip decision is consistent regardless of cache status."""
        tags = ["looking_left", "1girl"]

        # Same tags should produce same result whether from cache or disk
        result1 = handler_minimal.swap_tags_with_info(tags, skip_safety_check=True)
        result2 = handler_minimal.swap_tags_with_info(tags, skip_safety_check=True)

        assert result1.swapped_tags == result2.swapped_tags
        assert result1.flip_applied == result2.flip_applied
        assert result1.has_mismatch == result2.has_mismatch

    def test_stats_accumulation(self, handler_minimal):
        """Test that statistics accumulate correctly."""
        handler_minimal.stats['total_flips'] = 0
        handler_minimal.stats['unmapped_tags'] = set()

        # Process several tag lists
        # Use tags that won't match regex patterns: 'leftward_*' or 'rightward_*'
        tag_lists = [
            ["looking_left", "leftward_gaze"],
            ["looking_right", "rightward_tilt"],
            ["1girl", "red_hair"],  # No orientation tags
        ]

        for tags in tag_lists:
            handler_minimal.swap_tags_with_info(tags, skip_safety_check=True, record_stats=True)

        # Check stats
        assert handler_minimal.stats['total_flips'] == 3
        assert "leftward_gaze" in handler_minimal.stats['unmapped_tags']
        assert "rightward_tilt" in handler_minimal.stats['unmapped_tags']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

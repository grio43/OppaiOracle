"""Priority scoring for dataset grading.

Computes review priority scores that downweight known edge cases
(traps, multi-subject, gender ambiguity) and upweight likely real problems.
"""

from typing import Dict, List, Tuple, Set

# Tags related to gender identification - these are expected to have
# high false positive rates on trap/crossdresser images
GENDER_TAGS: Set[str] = {
    '1boy', '1girl', '2boys', '2girls', '3boys', '3girls',
    'multiple_boys', 'multiple_girls',
    'male_focus', 'female_focus',
    'trap', 'otoko_no_ko', 'josou_seme',
    'crossdressing', 'crossdresser',
    'androgynous', 'ambiguous_gender',
    'futanari', 'newhalf',
    'male', 'female',
}

# Tags for character counting - problematic with multi-subject images
COUNT_TAGS: Set[str] = {
    'solo', 'duo', 'trio', 'group',
    '1boy', '1girl', '2boys', '2girls', '3boys', '3girls',
    '4boys', '4girls', '5boys', '5girls',
    '6+boys', '6+girls',
    'multiple_boys', 'multiple_girls',
    'everyone', 'crowd',
}

# High-frequency thresholds for tag importance
VERY_COMMON_THRESHOLD = 100000  # Tags appearing 100k+ times
COMMON_THRESHOLD = 10000         # Tags appearing 10k+ times


def compute_priority(
    fp_tags: List[Tuple[str, float]],
    fn_tags: List[Tuple[str, float]],
    tag_frequencies: Dict[str, int],
) -> Tuple[float, bool, bool]:
    """Compute review priority score for an image.

    Higher priority = more likely a real labeling problem.
    Lower priority = likely a semantic edge case (trap, multi-subject, etc.)

    Args:
        fp_tags: List of (tag_name, confidence) for false positives
        fn_tags: List of (tag_name, confidence) for false negatives
        tag_frequencies: Dict mapping tag names to occurrence counts

    Returns:
        Tuple of (priority_score, has_gender_error, has_count_error)
    """
    score = 0.0
    has_gender_error = False
    has_count_error = False

    # Process false positives (model predicted, not in ground truth)
    for tag, conf in fp_tags:
        freq = tag_frequencies.get(tag, 0)
        is_gender = tag in GENDER_TAGS
        is_count = tag in COUNT_TAGS

        if is_gender:
            has_gender_error = True
        if is_count:
            has_count_error = True

        if is_gender or is_count:
            # Heavily downweight - likely semantic edge case
            # Still contribute a tiny amount so images with ONLY these
            # errors don't get priority 0
            score += 0.1 * conf
        elif freq >= VERY_COMMON_THRESHOLD:
            # Very common tag, model knows it well
            # Disagreement likely indicates real problem
            score += 2.0 * conf
        elif freq >= COMMON_THRESHOLD:
            # Common tag, moderate confidence in model
            score += 1.0 * conf
        else:
            # Rare tag - model might just be undertrained
            score += 0.3 * conf

    # Process false negatives (in ground truth, not predicted)
    for tag, conf in fn_tags:
        freq = tag_frequencies.get(tag, 0)
        is_gender = tag in GENDER_TAGS
        is_count = tag in COUNT_TAGS

        if is_gender:
            has_gender_error = True
        if is_count:
            has_count_error = True

        if is_gender or is_count:
            # Heavily downweight
            score += 0.1 * (1.0 - conf)  # Use inverse confidence for FN
        elif freq >= VERY_COMMON_THRESHOLD:
            # Very common tag should be predicted if present
            # High-frequency FN is concerning
            score += 1.5 * (1.0 - conf)
        elif freq >= COMMON_THRESHOLD:
            score += 0.8 * (1.0 - conf)
        else:
            # Rare tag FN - model might not have learned it yet
            score += 0.2 * (1.0 - conf)

    return score, has_gender_error, has_count_error


def is_gender_only_error(
    fp_tags: List[Tuple[str, float]],
    fn_tags: List[Tuple[str, float]],
) -> bool:
    """Check if all errors are gender/count related.

    These images are likely edge cases (traps, multi-subject)
    where the model's visual predictions are correct but
    the semantic labels differ.
    """
    all_tags = [tag for tag, _ in fp_tags] + [tag for tag, _ in fn_tags]

    if not all_tags:
        return False

    gender_count_tags = GENDER_TAGS | COUNT_TAGS
    return all(tag in gender_count_tags for tag in all_tags)


def categorize_error(tag: str) -> str:
    """Categorize an error tag for reporting.

    Returns one of: 'gender', 'count', 'common', 'rare'
    """
    if tag in GENDER_TAGS:
        return 'gender'
    if tag in COUNT_TAGS:
        return 'count'
    return 'other'

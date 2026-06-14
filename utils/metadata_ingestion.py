# utils/metadata_ingestion.py

"""
Common utilities for ingesting image metadata files.

Functions here convert a raw “tags” field into a list of tag tokens.
These helpers perform *no* normalisation on punctuation or case;
upstream code must preserve tags exactly.
"""

from typing import List

def parse_tags_field(tags_field) -> List[str]:
    """Turn a raw tags field into a list of tokens.

    Accepts None, string or iterable.  String values are split on
    commas, with whitespace stripped from each tag.  Empty or unsupported
    types yield an empty list.
    """
    if tags_field is None:
        return []
    if isinstance(tags_field, str):
        return [p.strip() for p in tags_field.split(',') if p.strip()]
    if isinstance(tags_field, (list, tuple)):
        return [str(t).strip() for t in tags_field if str(t).strip()]
    return []

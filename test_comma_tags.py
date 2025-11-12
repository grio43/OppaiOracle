#!/usr/bin/env python3
"""Test script to verify comma-separated tag parsing."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.metadata_ingestion import parse_tags_field

def test_parse_tags():
    """Test tag parsing with various inputs."""

    print("Testing comma-separated tag parsing...\n")

    # Test 1: Basic comma-separated tags
    test1 = "1girl, green_eyes"
    result1 = parse_tags_field(test1)
    print(f"Test 1: '{test1}'")
    print(f"Result: {result1}")
    print(f"Expected: ['1girl', 'green_eyes']")
    print(f"Pass: {result1 == ['1girl', 'green_eyes']}\n")

    # Test 2: Tags with extra whitespace
    test2 = "1girl,  green_eyes,   long_hair"
    result2 = parse_tags_field(test2)
    print(f"Test 2: '{test2}'")
    print(f"Result: {result2}")
    print(f"Expected: ['1girl', 'green_eyes', 'long_hair']")
    print(f"Pass: {result2 == ['1girl', 'green_eyes', 'long_hair']}\n")

    # Test 3: Tags without spaces after commas
    test3 = "1girl,green_eyes,long_hair"
    result3 = parse_tags_field(test3)
    print(f"Test 3: '{test3}'")
    print(f"Result: {result3}")
    print(f"Expected: ['1girl', 'green_eyes', 'long_hair']")
    print(f"Pass: {result3 == ['1girl', 'green_eyes', 'long_hair']}\n")

    # Test 4: Tags with trailing comma
    test4 = "1girl, green_eyes, "
    result4 = parse_tags_field(test4)
    print(f"Test 4: '{test4}'")
    print(f"Result: {result4}")
    print(f"Expected: ['1girl', 'green_eyes']")
    print(f"Pass: {result4 == ['1girl', 'green_eyes']}\n")

    # Test 5: List input (should also work)
    test5 = ["1girl", "green_eyes", "long_hair"]
    result5 = parse_tags_field(test5)
    print(f"Test 5: {test5}")
    print(f"Result: {result5}")
    print(f"Expected: ['1girl', 'green_eyes', 'long_hair']")
    print(f"Pass: {result5 == ['1girl', 'green_eyes', 'long_hair']}\n")

    # Test 6: Empty string
    test6 = ""
    result6 = parse_tags_field(test6)
    print(f"Test 6: '{test6}'")
    print(f"Result: {result6}")
    print(f"Expected: []")
    print(f"Pass: {result6 == []}\n")

    # Test 7: None input
    test7 = None
    result7 = parse_tags_field(test7)
    print(f"Test 7: {test7}")
    print(f"Result: {result7}")
    print(f"Expected: []")
    print(f"Pass: {result7 == []}\n")

    print("All tests completed!")

if __name__ == "__main__":
    test_parse_tags()

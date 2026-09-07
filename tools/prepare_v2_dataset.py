#!/usr/bin/env python3
"""Run only after all V2 subsets are ready. Does not launch training."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Configuration_System import load_config
from utils.v2_preparation import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_config(args.config)
    result = prepare_dataset(config)
    logging.info('Prepared %s training + %s validation images; %s vocabulary entries',
                 result['train_count'], result['validation_count'], result['vocabulary_size'])


if __name__ == '__main__':
    main()

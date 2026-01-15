#!/usr/bin/env python3
"""Analyze code review issues and categorize by status and severity."""

import os
import re
from pathlib import Path

def analyze_issues():
    issues_dir = Path('.code-review/issues')
    new_issues = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}

    for f in sorted(issues_dir.glob('CR-*.md')):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            status_match = re.search(r'^status:\s*(\w+)', content, re.MULTILINE)
            severity_match = re.search(r'^severity:\s*(\w+)', content, re.MULTILINE)

            if status_match and severity_match:
                status = status_match.group(1)
                severity = severity_match.group(1)
                if status == 'NEW':
                    new_issues.get(severity, []).append(f.stem)

    print("NEW Issues Summary:")
    print("=" * 60)
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if new_issues[severity]:
            print(f'\n{severity}: {len(new_issues[severity])} issues')
            print('  ' + ', '.join(new_issues[severity]))

    print(f"\n\nTotal NEW issues: {sum(len(v) for v in new_issues.values())}")

if __name__ == '__main__':
    analyze_issues()

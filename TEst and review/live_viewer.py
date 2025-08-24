#!/usr/bin/env python3
"""
Live viewer for monitoring evaluation results in real-time.
Supports both terminal (rich) and web (Flask) interfaces.
"""

import json
import time
import os
import sys
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Import vocabulary verification
import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent))
from vocabulary import verify_vocabulary_integrity

# Terminal UI version using rich
def extract_tags_from_result(result):
    """Extract tags from result, supporting both new and legacy formats."""
    # Try new schema first
    if 'tags' in result and isinstance(result['tags'], list):
        # New schema: tags is a list of {name, score} dicts
        return {tag['name'] for tag in result['tags'] \
                if not (tag['name'].startswith('tag_') and \
                       len(tag['name']) > 4 and \
                       tag['name'][4:].isdigit())}
    # Fall back to legacy schema
    elif 'predicted_tags' in result:
        return {tag for tag in result.get('predicted_tags', [])
                if not (tag.startswith('tag_') and len(tag) > 4 and tag[4:].isdigit())}
    else:
        return set()

def run_terminal_viewer(jsonl_file: str, refresh_interval: float = 1.0):
    """Terminal-based live viewer using rich library"""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.text import Text
        from rich.align import Align
    except ImportError:
        print("Please install 'rich' library: pip install rich")
        sys.exit(1)
    
    console = Console()
    
    class ResultsMonitor:
        def __init__(self, filepath):
            self.filepath = Path(filepath)
            self.last_position = 0
            self.total_lines = 0
            self.metrics_history = defaultdict(list)
            self.recent_results = deque(maxlen=10)
            self.tag_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            self.start_time = time.time()
            self.last_update = None
            
            # Running statistics
            self.stats = {
                'total_processed': 0,
                'avg_precision': 0,
                'avg_recall': 0,
                'avg_f1': 0,
                'perfect_matches': 0,
                'zero_f1_count': 0,
                'processing_rate': 0,
            }
            
            # F1 distribution bins
            self.f1_bins = {
                'excellent': 0,  # >0.8
                'good': 0,       # 0.6-0.8
                'fair': 0,       # 0.4-0.6
                'poor': 0,       # <0.4
            }
        
        def update(self):
            """Read new lines from file and update statistics"""
            if not self.filepath.exists():
                return False
            
            file_size = self.filepath.stat().st_size
            if file_size < self.last_position:
                # File was truncated/restarted
                self.last_position = 0
                self.reset_stats()
            
            new_lines = []
            with open(self.filepath, 'r') as f:
                f.seek(self.last_position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            new_lines.append(data)
                        except json.JSONDecodeError:
                            continue
                self.last_position = f.tell()
            
            if new_lines:
                self.process_new_results(new_lines)
                self.last_update = datetime.now()
                return True
            return False
        
        def reset_stats(self):
            """Reset all statistics"""
            self.metrics_history.clear()
            self.recent_results.clear()
            self.tag_performance.clear()
            self.stats = {k: 0 for k in self.stats}
            self.f1_bins = {k: 0 for k in self.f1_bins}
            self.total_lines = 0
        
        def process_new_results(self, results: List[Dict]):
            """Process new result entries"""
            for result in results:
                self.total_lines += 1
                metrics = result['metrics']
                
                # Update metrics history
                self.metrics_history['precision'].append(metrics['precision'])
                self.metrics_history['recall'].append(metrics['recall'])
                self.metrics_history['f1'].append(metrics['f1'])
                
                # Update F1 distribution
                f1 = metrics['f1']
                if f1 > 0.8:
                    self.f1_bins['excellent'] += 1
                elif f1 > 0.6:
                    self.f1_bins['good'] += 1
                elif f1 > 0.4:
                    self.f1_bins['fair'] += 1
                else:
                    self.f1_bins['poor'] += 1
                
                # Count perfect matches and failures
                if f1 == 1.0:
                    self.stats['perfect_matches'] += 1
                elif f1 == 0.0:
                    self.stats['zero_f1_count'] += 1
                
                # Update tag performance
                predicted = extract_tags_from_result(result)
                # Support both formats for ground truth
                ground_truth = set(result.get('ground_truth_tags', []))
                
                for tag in predicted & ground_truth:
                    self.tag_performance[tag]['tp'] += 1
                for tag in predicted - ground_truth:
                    self.tag_performance[tag]['fp'] += 1
                for tag in ground_truth - predicted:
                    self.tag_performance[tag]['fn'] += 1
                
                # Add to recent results
                self.recent_results.append({
                    'filename': result.get('filename', 'unknown')[:40],
                    'f1': f1,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'n_pred': result.get('num_pred_tags', 0),
                    'n_gt': result.get('num_gt_tags', 0),
                })
            
            # Update running averages
            self.stats['total_processed'] = self.total_lines
            if self.metrics_history['precision']:
                self.stats['avg_precision'] = np.mean(self.metrics_history['precision'])
                self.stats['avg_recall'] = np.mean(self.metrics_history['recall'])
                self.stats['avg_f1'] = np.mean(self.metrics_history['f1'])
            
            # Calculate processing rate
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.stats['processing_rate'] = self.total_lines / elapsed
        
        def get_top_tags(self, n=5, worst=False):
            """Get best or worst performing tags"""
            tag_scores = []
            for tag, counts in self.tag_performance.items():
                tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)
                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                support = tp + fn
                if support > 0:  # Only include tags with some support
                    tag_scores.append((tag, f1, support))
            
            tag_scores.sort(key=lambda x: x[1], reverse=not worst)
            return tag_scores[:n]
        
        def create_display(self) -> Layout:
            """Create the display layout"""
            layout = Layout()
            
            # Create header
            header = Panel(
                Align.center(
                    Text(f"🔍 Live Evaluation Monitor - {self.filepath.name}", 
                         style="bold cyan"),
                    vertical="middle"
                ),
                height=3
            )
            
            # Create stats panel
            stats_table = Table(show_header=False, box=None, padding=(0, 1))
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            elapsed = time.time() - self.start_time
            stats_table.add_row("Total Processed:", f"{self.stats['total_processed']:,}")
            stats_table.add_row("Processing Rate:", f"{self.stats['processing_rate']:.1f} img/s")
            stats_table.add_row("Elapsed Time:", f"{elapsed:.1f}s")
            stats_table.add_row("", "")
            stats_table.add_row("Avg Precision:", f"{self.stats['avg_precision']:.4f}")
            stats_table.add_row("Avg Recall:", f"{self.stats['avg_recall']:.4f}")
            stats_table.add_row("Avg F1 Score:", f"{self.stats['avg_f1']:.4f}")
            stats_table.add_row("", "")
            stats_table.add_row("Perfect Matches:", f"{self.stats['perfect_matches']:,}")
            stats_table.add_row("Zero F1 Count:", f"{self.stats['zero_f1_count']:,}")
            
            stats_panel = Panel(stats_table, title="📊 Overall Statistics", border_style="green")
            
            # Create F1 distribution panel
            dist_table = Table(show_header=True, box=None)
            dist_table.add_column("Range", style="cyan", width=15)
            dist_table.add_column("Count", style="white", width=10)
            dist_table.add_column("Percentage", style="yellow", width=10)
            
            total = sum(self.f1_bins.values()) or 1
            dist_table.add_row(
                "Excellent (>0.8)", 
                f"{self.f1_bins['excellent']:,}",
                f"{100*self.f1_bins['excellent']/total:.1f}%"
            )
            dist_table.add_row(
                "Good (0.6-0.8)", 
                f"{self.f1_bins['good']:,}",
                f"{100*self.f1_bins['good']/total:.1f}%"
            )
            dist_table.add_row(
                "Fair (0.4-0.6)", 
                f"{self.f1_bins['fair']:,}",
                f"{100*self.f1_bins['fair']/total:.1f}%"
            )
            dist_table.add_row(
                "Poor (<0.4)", 
                f"{self.f1_bins['poor']:,}",
                f"{100*self.f1_bins['poor']/total:.1f}%"
            )
            
            dist_panel = Panel(dist_table, title="📈 F1 Score Distribution", border_style="blue")
            
            # Create recent results panel
            recent_table = Table(show_header=True, box=None)
            recent_table.add_column("File", style="cyan", width=30)
            recent_table.add_column("F1", style="white", width=6)
            recent_table.add_column("Prec", style="green", width=6)
            recent_table.add_column("Rec", style="blue", width=6)
            recent_table.add_column("Tags", style="yellow", width=10)
            
            for r in reversed(self.recent_results):
                # Color code F1 scores
                f1 = r['f1']
                if f1 > 0.8:
                    f1_style = "green"
                elif f1 > 0.6:
                    f1_style = "yellow"
                else:
                    f1_style = "red"
                
                recent_table.add_row(
                    r['filename'][:30],
                    Text(f"{f1:.3f}", style=f1_style),
                    f"{r['precision']:.3f}",
                    f"{r['recall']:.3f}",
                    f"{r['n_pred']}/{r['n_gt']}"
                )
            
            recent_panel = Panel(recent_table, title="📝 Recent Results", border_style="yellow")
            
            # Create tag performance panels
            best_tags = self.get_top_tags(5, worst=False)
            worst_tags = self.get_top_tags(5, worst=True)
            
            best_table = Table(show_header=True, box=None)
            best_table.add_column("Tag", style="cyan", width=20)
            best_table.add_column("F1", style="green", width=6)
            best_table.add_column("Support", style="white", width=8)
            
            for tag, f1, support in best_tags:
                best_table.add_row(tag[:20], f"{f1:.3f}", str(support))
            
            best_panel = Panel(best_table, title="✅ Best Tags", border_style="green")
            
            worst_table = Table(show_header=True, box=None)
            worst_table.add_column("Tag", style="cyan", width=20)
            worst_table.add_column("F1", style="red", width=6)
            worst_table.add_column("Support", style="white", width=8)
            
            for tag, f1, support in worst_tags:
                worst_table.add_row(tag[:20], f"{f1:.3f}", str(support))
            
            worst_panel = Panel(worst_table, title="❌ Worst Tags", border_style="red")
            
            # Arrange layout
            layout.split_column(
                header,
                Layout(name="main", ratio=10)
            )
            
            layout["main"].split_row(
                Layout(name="left", ratio=1),
                Layout(name="middle", ratio=1),
                Layout(name="right", ratio=1)
            )
            
            layout["left"].split_column(
                stats_panel,
                dist_panel
            )
            
            layout["middle"].split_column(
                recent_panel
            )
            
            layout["right"].split_column(
                best_panel,
                worst_panel
            )
            
            return layout
    
    # Initialize monitor
    monitor = ResultsMonitor(jsonl_file)
    
    # Create live display
    with Live(monitor.create_display(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                if monitor.update():
                    live.update(monitor.create_display())
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")
            
            # Print final summary
            console.print("\n[bold cyan]Final Statistics:[/bold cyan]")
            console.print(f"Total Processed: {monitor.stats['total_processed']:,}")
            console.print(f"Average F1 Score: {monitor.stats['avg_f1']:.4f}")
            console.print(f"Average Precision: {monitor.stats['avg_precision']:.4f}")
            console.print(f"Average Recall: {monitor.stats['avg_recall']:.4f}")


# Web-based viewer using Flask
def run_web_viewer(jsonl_file: str, port: int = 5000):
    """Web-based live viewer using Flask"""
    try:
        from flask import Flask, render_template_string, jsonify
    except ImportError:
        print("Please install Flask: pip install flask")
        sys.exit(1)
    
    app = Flask(__name__)
    
    # HTML template with auto-refresh
    HTML_TEMPLATE = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Evaluation Monitor</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 20px;
                background: #1e1e1e;
                color: #e0e0e0;
            }
            h1 {
                color: #4fc3f7;
                text-align: center;
                margin-bottom: 30px;
            }
            .container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }
            .panel {
                background: #2a2a2a;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }
            .panel h2 {
                color: #81c784;
                margin-top: 0;
                font-size: 1.2em;
                border-bottom: 2px solid #81c784;
                padding-bottom: 10px;
            }
            .stat {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 5px 0;
                border-bottom: 1px solid #3a3a3a;
            }
            .stat-label {
                color: #a0a0a0;
            }
            .stat-value {
                font-weight: bold;
                color: #ffffff;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background: #3a3a3a;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4caf50, #81c784);
                transition: width 0.3s ease;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th {
                background: #3a3a3a;
                padding: 8px;
                text-align: left;
                color: #81c784;
            }
            td {
                padding: 8px;
                border-bottom: 1px solid #3a3a3a;
            }
            .good { color: #4caf50; }
            .fair { color: #ffa726; }
            .poor { color: #ef5350; }
            .timestamp {
                text-align: center;
                color: #7a7a7a;
                margin-top: 20px;
                font-size: 0.9em;
            }
        </style>
        <script>
            // Auto-refresh data via AJAX
            setInterval(function() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        // Update would go here for smoother updates
                        // For now, letting meta refresh handle it
                    });
            }, 1000);
        </script>
    </head>
    <body>
        <h1>🔍 Live Evaluation Monitor</h1>
        <div class="container">
            <div class="panel">
                <h2>📊 Overall Statistics</h2>
                <div class="stat">
                    <span class="stat-label">Total Processed:</span>
                    <span class="stat-value">{{ stats.total_processed|default(0) }}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Processing Rate:</span>
                    <span class="stat-value">{{ "%.1f"|format(stats.processing_rate|default(0)) }} img/s</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg F1 Score:</span>
                    <span class="stat-value">{{ "%.4f"|format(stats.avg_f1|default(0)) }}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Precision:</span>
                    <span class="stat-value">{{ "%.4f"|format(stats.avg_precision|default(0)) }}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Recall:</span>
                    <span class="stat-value">{{ "%.4f"|format(stats.avg_recall|default(0)) }}</span>
                </div>
            </div>
            
            <div class="panel">
                <h2>📈 F1 Score Distribution</h2>
                {% for category, count in f1_dist.items() %}
                <div class="stat">
                    <span class="stat-label">{{ category }}:</span>
                    <span class="stat-value">{{ count }} ({{ "%.1f"|format(percentages[category]) }}%)</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ percentages[category] }}%"></div>
                </div>
                {% endfor %}
            </div>
            
            <div class="panel">
                <h2>📝 Recent Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>F1</th>
                            <th>Prec</th>
                            <th>Rec</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in recent_results %}
                        <tr>
                            <td>{{ result.filename[:25] }}...</td>
                            <td class="{% if result.f1 > 0.8 %}good{% elif result.f1 > 0.6 %}fair{% else %}poor{% endif %}">
                                {{ "%.3f"|format(result.f1) }}
                            </td>
                            <td>{{ "%.3f"|format(result.precision) }}</td>
                            <td>{{ "%.3f"|format(result.recall) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="panel">
                <h2>🏆 Performance Leaders</h2>
                <h3 style="color: #4caf50;">Best Tags</h3>
                <table>
                    <thead>
                        <tr><th>Tag</th><th>F1</th></tr>
                    </thead>
                    <tbody>
                        {% for tag, f1, support in best_tags %}
                        <tr>
                            <td>{{ tag[:20] }}</td>
                            <td class="good">{{ "%.3f"|format(f1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                
                <h3 style="color: #ef5350; margin-top: 20px;">Worst Tags</h3>
                <table>
                    <thead>
                        <tr><th>Tag</th><th>F1</th></tr>
                    </thead>
                    <tbody>
                        {% for tag, f1, support in worst_tags %}
                        <tr>
                            <td>{{ tag[:20] }}</td>
                            <td class="poor">{{ "%.3f"|format(f1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        <div class="timestamp">Last Updated: {{ timestamp }}</div>
    </body>
    </html>
    '''
    
    class WebMonitor:
        def __init__(self, filepath):
            self.filepath = Path(filepath)
            self.last_position = 0
            self.stats = {
                'total_processed': 0,
                'avg_precision': 0,
                'avg_recall': 0,
                'avg_f1': 0,
                'processing_rate': 0,
            }
            self.f1_bins = {
                'Excellent (>0.8)': 0,
                'Good (0.6-0.8)': 0,
                'Fair (0.4-0.6)': 0,
                'Poor (<0.4)': 0,
            }
            self.recent_results = deque(maxlen=10)
            self.tag_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            self.metrics_history = defaultdict(list)
            self.start_time = time.time()
        
        def update(self):
            if not self.filepath.exists():
                return
            
            with open(self.filepath, 'r') as f:
                f.seek(self.last_position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self.process_result(data)
                        except json.JSONDecodeError:
                            continue
                self.last_position = f.tell()
            
            # Update averages
            if self.metrics_history['f1']:
                self.stats['avg_precision'] = np.mean(self.metrics_history['precision'])
                self.stats['avg_recall'] = np.mean(self.metrics_history['recall'])
                self.stats['avg_f1'] = np.mean(self.metrics_history['f1'])
            
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.stats['processing_rate'] = self.stats['total_processed'] / elapsed
        
        def process_result(self, result):
            self.stats['total_processed'] += 1
            metrics = result['metrics']
            
            # Update history
            self.metrics_history['precision'].append(metrics['precision'])
            self.metrics_history['recall'].append(metrics['recall'])
            self.metrics_history['f1'].append(metrics['f1'])
            
            # Update F1 bins
            f1 = metrics['f1']
            if f1 > 0.8:
                self.f1_bins['Excellent (>0.8)'] += 1
            elif f1 > 0.6:
                self.f1_bins['Good (0.6-0.8)'] += 1
            elif f1 > 0.4:
                self.f1_bins['Fair (0.4-0.6)'] += 1
            else:
                self.f1_bins['Poor (<0.4)'] += 1
            
            # Update recent
            self.recent_results.append({
                'filename': result.get('filename', 'unknown'),
                'f1': f1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
            })
            
            # Update tag performance
            predicted = extract_tags_from_result(result)
            # Support both formats for ground truth
            ground_truth = set(result.get('ground_truth_tags', []))
            
            for tag in predicted & ground_truth:
                self.tag_performance[tag]['tp'] += 1
            for tag in predicted - ground_truth:
                self.tag_performance[tag]['fp'] += 1
            for tag in ground_truth - predicted:
                self.tag_performance[tag]['fn'] += 1
        
        def get_top_tags(self, n=5, worst=False):
            tag_scores = []
            for tag, counts in self.tag_performance.items():
                tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                support = tp + fn
                if support > 0:
                    tag_scores.append((tag, f1, support))
            
            tag_scores.sort(key=lambda x: x[1], reverse=not worst)
            return tag_scores[:n]
    
    monitor = WebMonitor(jsonl_file)
    
    @app.route('/')
    def index():
        monitor.update()
        
        # Calculate percentages
        total = sum(monitor.f1_bins.values()) or 1
        percentages = {k: 100 * v / total for k, v in monitor.f1_bins.items()}
        
        return render_template_string(
            HTML_TEMPLATE,
            stats=monitor.stats,
            f1_dist=monitor.f1_bins,
            percentages=percentages,
            recent_results=list(reversed(monitor.recent_results)),
            best_tags=monitor.get_top_tags(5, worst=False),
            worst_tags=monitor.get_top_tags(5, worst=True),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    @app.route('/api/stats')
    def api_stats():
        monitor.update()
        return jsonify({
            'stats': monitor.stats,
            'f1_distribution': monitor.f1_bins,
            'total_processed': monitor.stats['total_processed']
        })
    
    print(f"Starting web server on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description='Live monitoring of evaluation results')
    parser.add_argument(
        'jsonl_file',
        help='Path to the JSONL results file to monitor'
    )
    parser.add_argument(
        '--mode',
        choices=['terminal', 'web'],
        default='terminal',
        help='Display mode (default: terminal)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for web server (default: 5000)'
    )
    parser.add_argument(
        '--refresh',
        type=float,
        default=1.0,
        help='Refresh interval in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists or will be created
    if not Path(args.jsonl_file).exists():
        print(f"Warning: File {args.jsonl_file} does not exist yet.")
        print("Waiting for file to be created...")
        while not Path(args.jsonl_file).exists():
            time.sleep(1)
        print("File detected! Starting monitor...")
    
    if args.mode == 'terminal':
        run_terminal_viewer(args.jsonl_file, args.refresh)
    else:
        run_web_viewer(args.jsonl_file, args.port)


if __name__ == '__main__':
    # Example usage if run directly
    if len(sys.argv) == 1:
        # Default to the file mentioned by the user
        default_file = '/media/andrewk/qnap-public/workspace/results/evaluation_results_gpu_results.jsonl'
        print(f"No arguments provided. Using default file: {default_file}")
        print("Run with --help for more options")
        print()
        run_terminal_viewer(default_file)
    else:
        main()
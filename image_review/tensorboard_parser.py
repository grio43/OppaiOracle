"""TensorBoard event file parser for image review."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class PredictionEntry:
    """Single tag prediction with status."""
    tag: str
    probability: float
    expected: bool  # True = ground truth tag
    status: str     # "TP", "FP", "FN"


@dataclass
class RatingInfo:
    """Rating prediction and ground truth."""
    predicted: str              # e.g., "safe", "explicit"
    predicted_confidence: float # e.g., 0.952
    actual: str                 # e.g., "explicit"
    is_correct: bool            # whether prediction matches actual


@dataclass
class SampleData:
    """Data for a single validation sample."""
    step: int
    sample_index: int
    image_data: bytes           # PNG encoded image
    predictions: List[PredictionEntry] = field(default_factory=list)
    ground_truth_tags: List[str] = field(default_factory=list)
    rating: Optional[RatingInfo] = None  # Rating info if available

    @property
    def tp_count(self) -> int:
        return sum(1 for p in self.predictions if p.status == "TP")

    @property
    def fp_count(self) -> int:
        return sum(1 for p in self.predictions if p.status == "FP")

    @property
    def fn_count(self) -> int:
        return sum(1 for p in self.predictions if p.status == "FN")


class TensorBoardParser:
    """Parse TensorBoard event files for validation samples."""

    def __init__(self, logdir: Path):
        self.logdir = Path(logdir)
        self._accumulator = None
        self._samples: Dict[Tuple[int, int], SampleData] = {}

    def load(self) -> bool:
        """Load event files from the log directory."""
        if not self.logdir.exists():
            return False

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            print("tensorboard package not installed")
            return False

        self._accumulator = EventAccumulator(str(self.logdir))
        self._accumulator.Reload()
        self._parse_samples()
        return True

    def _parse_samples(self):
        """Parse all validation samples from events."""
        if self._accumulator is None:
            return

        tags = self._accumulator.Tags()
        image_tags = tags.get('images', [])

        # Find image tags matching pattern: {prefix}/sample_{i}/image
        sample_image_tags = [t for t in image_tags if '/sample_' in t and t.endswith('/image')]

        for img_tag in sample_image_tags:
            # Extract prefix and sample index
            # e.g., "val/sample_0/image" -> prefix="val", idx=0
            match = re.match(r'(.+)/sample_(\d+)/image', img_tag)
            if not match:
                continue

            prefix = match.group(1)
            sample_idx = int(match.group(2))
            # TensorBoard stores add_text() as tensors with /text_summary suffix
            text_tag = f"{prefix}/sample_{sample_idx}/topk/text_summary"

            # Get all steps for this sample's images
            try:
                img_events = self._accumulator.Images(img_tag)
            except KeyError:
                continue

            for img_event in img_events:
                step = img_event.step

                # Decode image
                image_data = img_event.encoded_image_string

                # Parse prediction table if available
                predictions = []
                ground_truth = []

                # Try to get text data (stored as tensors in newer TensorBoard)
                rating_info = None
                try:
                    # First try tensors (newer format)
                    text_events = self._accumulator.Tensors(text_tag)
                    for text_event in text_events:
                        if text_event.step == step:
                            # Extract string from tensor
                            tensor_proto = text_event.tensor_proto
                            if hasattr(tensor_proto, 'string_val') and tensor_proto.string_val:
                                markdown = tensor_proto.string_val[0].decode('utf-8')
                            else:
                                # Try to decode from tensor content
                                markdown = tensor_proto.tensor_content.decode('utf-8', errors='ignore')
                            predictions, ground_truth, rating_info = self._parse_markdown_table(markdown)
                            break
                except (KeyError, AttributeError, IndexError):
                    pass

                # Fallback: try text summaries (older format)
                if not predictions:
                    try:
                        text_events = self._accumulator.Texts(text_tag)
                        for text_event in text_events:
                            if text_event.step == step:
                                markdown = text_event.value
                                predictions, ground_truth, rating_info = self._parse_markdown_table(markdown)
                                break
                    except (KeyError, AttributeError):
                        pass

                sample = SampleData(
                    step=step,
                    sample_index=sample_idx,
                    image_data=image_data,
                    predictions=predictions,
                    ground_truth_tags=ground_truth,
                    rating=rating_info
                )
                self._samples[(step, sample_idx)] = sample

    def _parse_markdown_table(self, markdown: str) -> Tuple[List[PredictionEntry], List[str], Optional[RatingInfo]]:
        """Parse markdown table from log_predictions.

        Format (with rating):
        **Rating:** safe (95.2%) | Actual: explicit | WRONG

        | tag | prob | expected | status |
        | --- | --- | --- | --- |
        | tag_name | 0.9876 | YES | TP |

        Format (without rating):
        | tag | prob | expected | status |
        | --- | --- | --- | --- |
        | tag_name | 0.9876 | YES | TP |
        """
        predictions = []
        ground_truth = []
        rating_info = None

        lines = markdown.strip().split('\n')

        # Check for rating line at the start
        # Format: **Rating:** safe (95.2%) | Actual: explicit | WRONG
        table_start = 0
        if lines and lines[0].startswith('**Rating:**'):
            rating_line = lines[0]
            # Parse: **Rating:** safe (95.2%) | Actual: explicit | CORRECT/WRONG
            rating_match = re.match(
                r'\*\*Rating:\*\*\s*(\w+)\s*\((\d+\.?\d*)%\)\s*\|\s*Actual:\s*(\w+)\s*\|\s*(\w+)',
                rating_line
            )
            if rating_match:
                pred_rating = rating_match.group(1).lower()
                pred_conf = float(rating_match.group(2)) / 100.0
                actual_rating = rating_match.group(3).lower()
                is_correct = rating_match.group(4).upper() == 'CORRECT'
                rating_info = RatingInfo(
                    predicted=pred_rating,
                    predicted_confidence=pred_conf,
                    actual=actual_rating,
                    is_correct=is_correct
                )
            # Skip rating line and empty line
            table_start = 2 if len(lines) > 1 and not lines[1].strip() else 1

        # Find the table header line
        header_idx = None
        for idx, line in enumerate(lines[table_start:], start=table_start):
            if '| tag |' in line.lower() or ('|' in line and 'prob' in line.lower()):
                header_idx = idx
                break

        if header_idx is None:
            return predictions, ground_truth, rating_info

        # Parse table rows (skip header and separator)
        for line in lines[header_idx + 2:]:
            # Split by | and filter empty parts
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                tag = parts[0]
                try:
                    prob = float(parts[1])
                except ValueError:
                    prob = 0.0
                expected = parts[2].upper() == 'YES'
                status = parts[3].upper()

                predictions.append(PredictionEntry(
                    tag=tag,
                    probability=prob,
                    expected=expected,
                    status=status
                ))

                if expected:
                    ground_truth.append(tag)

        return predictions, ground_truth, rating_info

    def get_available_steps(self) -> List[int]:
        """Get all steps with validation data."""
        return sorted(set(step for step, _ in self._samples.keys()))

    def get_samples_at_step(self, step: int) -> List[SampleData]:
        """Get all samples at a given step."""
        return sorted(
            [s for (st, _), s in self._samples.items() if st == step],
            key=lambda s: s.sample_index
        )

    def get_sample(self, step: int, sample_idx: int) -> Optional[SampleData]:
        """Get a specific sample."""
        return self._samples.get((step, sample_idx))

    def get_all_samples(self) -> List[SampleData]:
        """Get all samples sorted by step then index."""
        return sorted(self._samples.values(), key=lambda s: (s.step, s.sample_index))

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        return len(self._samples)


def discover_runs(tensorboard_root: Path) -> List[Dict[str, Any]]:
    """Discover all TensorBoard run directories."""
    runs = []
    tensorboard_root = Path(tensorboard_root)

    if not tensorboard_root.exists():
        return runs

    for run_dir in tensorboard_root.iterdir():
        if run_dir.is_dir():
            # Check if this directory contains event files
            event_files = list(run_dir.rglob('events.out.tfevents.*'))
            if event_files:
                # Parse timestamp from directory name (format: YYYYMMDD-HHMMSS-hostname)
                name = run_dir.name
                parts = name.split('-')
                timestamp = parts[0] if parts else name

                runs.append({
                    'name': name,
                    'path': str(run_dir),
                    'timestamp': timestamp
                })

    # Sort by timestamp descending (newest first)
    return sorted(runs, key=lambda r: r['timestamp'], reverse=True)

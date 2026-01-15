"""FastAPI server for image review."""

from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .tensorboard_parser import TensorBoardParser, discover_runs, PredictionEntry, RatingInfo


class PredictionModel(BaseModel):
    """Prediction entry for API response."""
    tag: str
    probability: float
    expected: bool
    status: str


class RatingModel(BaseModel):
    """Rating prediction for API response."""
    predicted: str
    predicted_confidence: float
    actual: str
    is_correct: bool


class SampleModel(BaseModel):
    """Sample data for API response."""
    step: int
    sample_index: int
    predictions: List[PredictionModel]
    ground_truth_tags: List[str]
    tp_count: int
    fp_count: int
    fn_count: int
    rating: Optional[RatingModel] = None


class NavigationModel(BaseModel):
    """Navigation info for API response."""
    current_index: int
    total_samples: int
    current_step: int
    current_sample_idx: int
    has_prev: bool
    has_next: bool


class StepInfo(BaseModel):
    """Step info for API response."""
    step: int
    sample_count: int


# Module-level state for current parser and run
_current_parser: Optional[TensorBoardParser] = None
_current_run: Optional[str] = None
_tensorboard_root: Optional[Path] = None


def create_app(tensorboard_root: Path = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _tensorboard_root

    app = FastAPI(title="Image Review Service")

    if tensorboard_root is None:
        tensorboard_root = Path(__file__).parent.parent / "tensorboard"

    _tensorboard_root = Path(tensorboard_root)

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main HTML page."""
        html_path = static_dir / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="index.html not found")
        return html_path.read_text(encoding='utf-8')

    @app.get("/api/runs")
    async def get_runs():
        """Get available TensorBoard runs."""
        runs = discover_runs(_tensorboard_root)
        return {"runs": runs}

    @app.post("/api/runs/{run_name}/load")
    async def load_run(run_name: str):
        """Load a TensorBoard run."""
        global _current_parser, _current_run

        run_path = _tensorboard_root / run_name
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        parser = TensorBoardParser(run_path)
        if not parser.load():
            raise HTTPException(status_code=500, detail="Failed to load run")

        _current_parser = parser
        _current_run = run_name

        steps = parser.get_available_steps()
        return {
            "success": True,
            "run": run_name,
            "total_samples": parser.get_sample_count(),
            "steps": [
                {"step": s, "sample_count": len(parser.get_samples_at_step(s))}
                for s in steps
            ]
        }

    @app.get("/api/steps")
    async def get_steps():
        """Get available validation steps."""
        if _current_parser is None:
            raise HTTPException(status_code=400, detail="No run loaded")

        steps = _current_parser.get_available_steps()
        return {
            "steps": [
                StepInfo(step=s, sample_count=len(_current_parser.get_samples_at_step(s)))
                for s in steps
            ]
        }

    @app.get("/api/samples")
    async def get_samples(
        step: Optional[int] = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500)
    ):
        """Get paginated list of samples."""
        if _current_parser is None:
            raise HTTPException(status_code=400, detail="No run loaded")

        if step is not None:
            samples = _current_parser.get_samples_at_step(step)
        else:
            samples = _current_parser.get_all_samples()

        total = len(samples)
        samples = samples[offset:offset + limit]

        def to_sample_model(s):
            rating = None
            if s.rating:
                rating = RatingModel(
                    predicted=s.rating.predicted,
                    predicted_confidence=s.rating.predicted_confidence,
                    actual=s.rating.actual,
                    is_correct=s.rating.is_correct
                )
            return SampleModel(
                step=s.step,
                sample_index=s.sample_index,
                predictions=[
                    PredictionModel(
                        tag=p.tag,
                        probability=p.probability,
                        expected=p.expected,
                        status=p.status
                    ) for p in s.predictions
                ],
                ground_truth_tags=s.ground_truth_tags,
                tp_count=s.tp_count,
                fp_count=s.fp_count,
                fn_count=s.fn_count,
                rating=rating
            )

        return {
            "samples": [to_sample_model(s) for s in samples],
            "total": total,
            "offset": offset,
            "limit": limit
        }

    @app.get("/api/samples/{step}/{sample_idx}")
    async def get_sample(step: int, sample_idx: int):
        """Get a specific sample."""
        if _current_parser is None:
            raise HTTPException(status_code=400, detail="No run loaded")

        sample = _current_parser.get_sample(step, sample_idx)
        if sample is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        # Calculate navigation
        all_samples = _current_parser.get_all_samples()
        current_index = next(
            (i for i, s in enumerate(all_samples)
             if s.step == step and s.sample_index == sample_idx),
            0
        )

        # Build rating model if available
        rating = None
        if sample.rating:
            rating = RatingModel(
                predicted=sample.rating.predicted,
                predicted_confidence=sample.rating.predicted_confidence,
                actual=sample.rating.actual,
                is_correct=sample.rating.is_correct
            )

        return {
            "sample": SampleModel(
                step=sample.step,
                sample_index=sample.sample_index,
                predictions=[
                    PredictionModel(
                        tag=p.tag,
                        probability=p.probability,
                        expected=p.expected,
                        status=p.status
                    ) for p in sample.predictions
                ],
                ground_truth_tags=sample.ground_truth_tags,
                tp_count=sample.tp_count,
                fp_count=sample.fp_count,
                fn_count=sample.fn_count,
                rating=rating
            ),
            "navigation": NavigationModel(
                current_index=current_index,
                total_samples=len(all_samples),
                current_step=step,
                current_sample_idx=sample_idx,
                has_prev=current_index > 0,
                has_next=current_index < len(all_samples) - 1
            )
        }

    @app.get("/api/samples/{step}/{sample_idx}/image")
    async def get_sample_image(step: int, sample_idx: int):
        """Get the image for a sample."""
        if _current_parser is None:
            raise HTTPException(status_code=400, detail="No run loaded")

        sample = _current_parser.get_sample(step, sample_idx)
        if sample is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        return Response(
            content=sample.image_data,
            media_type="image/png"
        )

    @app.get("/api/navigate/{direction}")
    async def navigate(direction: str, current_step: int, current_idx: int):
        """Navigate to prev/next sample."""
        if _current_parser is None:
            raise HTTPException(status_code=400, detail="No run loaded")

        all_samples = _current_parser.get_all_samples()
        current_index = next(
            (i for i, s in enumerate(all_samples)
             if s.step == current_step and s.sample_index == current_idx),
            0
        )

        if direction == "prev" and current_index > 0:
            new_sample = all_samples[current_index - 1]
        elif direction == "next" and current_index < len(all_samples) - 1:
            new_sample = all_samples[current_index + 1]
        else:
            raise HTTPException(status_code=400, detail="Cannot navigate in that direction")

        return {
            "step": new_sample.step,
            "sample_idx": new_sample.sample_index
        }

    @app.get("/api/sample-by-index/{index}")
    async def get_sample_by_index(index: int):
        """Get sample by global index (0-based)."""
        if _current_parser is None:
            raise HTTPException(status_code=400, detail="No run loaded")

        all_samples = _current_parser.get_all_samples()
        if index < 0 or index >= len(all_samples):
            raise HTTPException(status_code=404, detail="Index out of range")

        sample = all_samples[index]
        return {
            "step": sample.step,
            "sample_idx": sample.sample_index
        }

    return app


def main():
    """Entry point for uvicorn."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='Image Review Server')
    parser.add_argument('--tensorboard-dir', type=str, default='./tensorboard',
                        help='Path to TensorBoard log directory')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    app = create_app(Path(args.tensorboard_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""
GET /v1/eval/results — serves the last eval harness run results.

The eval script writes results.json to tests/eval/results.json.
This endpoint reads that file so the README can link to a live URL.
Returns 404 if the eval has not been run yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(tags=["eval"])

# Relative to api/ directory (where uvicorn is launched from)
RESULTS_PATH = Path(__file__).parent.parent.parent / "tests" / "eval" / "results.json"


@router.get("/v1/eval/results")
async def get_eval_results() -> JSONResponse:
    """Return the latest eval harness results."""
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Eval results not found. "
                "Run: uv run python tests/eval/run_eval.py --document-id <uuid>"
            ),
        )

    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {e}")

    modes = data.get("modes", {})
    summary = {
        "run_at": data.get("run_at"),
        "eval_set": data.get("eval_set"),
        "document_id": data.get("document_id"),
        "abstention_rate": data.get("abstention_rate"),
        "metrics": {
            "recall_at_5": {
                mode: info.get("recall_at_5") for mode, info in modes.items()
            },
            "answer_accuracy_hybrid": modes.get("hybrid", {}).get("answer_accuracy"),
            "citation_precision_hybrid": modes.get("hybrid", {}).get("citation_precision"),
        },
        "full": data,
    }

    return JSONResponse(content=summary)

﻿import json
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from models import get_model
from train import MODELS_DIR, OUTPUT_DIR, train_model

app = FastAPI(title="Fake News ML Service")

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

training_status: Dict[str, Dict] = {}


class TrainRequest(BaseModel):
    model_name: str = "roberta-base"
    download_dataset: bool = False
    force_rebuild: bool = False


class PredictRequest(BaseModel):
    model_name: str = "roberta-base"
    text: str


def _model_directory(model_name: str) -> Path:
    return MODELS_DIR / model_name


def _model_is_trained(model_name: str) -> bool:
    model_dir = _model_directory(model_name)
    return (model_dir / "config.json").exists()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    models = [
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    ]
    return {"models": sorted(models)}


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    model_name = request.model_name
    status = training_status.get(model_name)
    if status and status.get("state") == "in_progress":
        raise HTTPException(status_code=400, detail="Training already in progress.")

    def run_training():
        training_status[model_name] = {"state": "in_progress"}
        try:
            summary = train_model(
                model_name=model_name,
                download_dataset=request.download_dataset,
                models_dir=MODELS_DIR,
                force_rebuild=request.force_rebuild,
            )
            training_status[model_name] = {
                "state": "completed",
                "summary": summary,
            }
        except Exception as exc:  # pylint: disable=broad-except
            training_status[model_name] = {
                "state": "failed",
                "error": str(exc),
            }

    background_tasks.add_task(run_training)
    return {"message": f"Training for model '{model_name}' started."}


@app.get("/train/status/{model_name}")
async def get_training_status(model_name: str):
    status = training_status.get(model_name)
    if not status:
        raise HTTPException(status_code=404, detail="No training job found.")

    return {"model_name": model_name, **status}


@app.post("/predict")
async def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    if not _model_is_trained(request.model_name):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' has not been trained yet.",
        )

    model = get_model(request.model_name, MODELS_DIR)
    return model.predict(request.text)


@app.get("/metrics/{model_name}")
async def get_metrics(model_name: str):
    metrics_path = OUTPUT_DIR / model_name / "classification_report.json"
    summary_path = OUTPUT_DIR / model_name / "summary.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for model '{model_name}'.",
        )

    with metrics_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

    return {
        "model_name": model_name,
        "report": report,
        "summary": summary,
    }


@app.get("/metrics/plots/{model_name}/{plot_name}")
async def get_plot(model_name: str, plot_name: str):
    if plot_name not in {"confusion_matrix.png", "roc_curve.png"}:
        raise HTTPException(status_code=404, detail="Plot not available.")

    plot_path = OUTPUT_DIR / model_name / plot_name
    if not plot_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Plot '{plot_name}' not found for model '{model_name}'.",
        )

    return FileResponse(plot_path)


from typing import List, Optional
from pydantic import BaseModel
from xai import shap_explain, lime_explain, integrated_gradients_explain
from projection import project_texts
import os

class ExplainRequest(BaseModel):
    text: str
    model_name: str
    method: str = "shap"
    top_k: int = 10

class ProjectRequest(BaseModel):
    texts: List[str]
    labels: Optional[List[str]] = None
    method: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    perplexity: float = 30.0
    random_state: int = 42

@app.post("/explain")
async def explain(request: ExplainRequest):
    if not _model_is_trained(request.model_name):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' has not been trained yet.")
    model = get_model(request.model_name, MODELS_DIR)
    tok = model.tokenizer
    m = request.method.lower()
    if m == "shap":
        return shap_explain(request.text, model.model, tok, top_k=request.top_k)
    if m == "lime":
        return lime_explain(request.text, model.model, tok, top_k=request.top_k)
    if m in {"ig","integrated_gradients"}:
        return integrated_gradients_explain(request.text, model.model, tok, top_k=request.top_k)
    raise HTTPException(status_code=400, detail="Unknown method")

@app.post("/project")
async def project(req: ProjectRequest):
    if not req.texts: raise HTTPException(status_code=400, detail="texts required")
    return project_texts(req.texts, labels=req.labels, method=req.method, n_neighbors=req.n_neighbors, min_dist=req.min_dist, perplexity=req.perplexity, random_state=req.random_state)

@app.get("/report/download/{model_name}")
async def report_download(model_name: str):
    folder = OUTPUT_DIR / model_name
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Model outputs not found.")
    zip_path = OUTPUT_DIR / f"{model_name}_report.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath,_,filenames in os.walk(folder):
            for f in filenames:
                fp = Path(dirpath) / f
                z.write(fp, fp.relative_to(OUTPUT_DIR))
    return FileResponse(str(zip_path), media_type="application/zip", filename=zip_path.name)

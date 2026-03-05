from fastapi import FastAPI

from configuration.rabbitmq_queues import XAI_REQUEST_QUEUE
from services import rabbitmq_service

from configuration.experiment_config import EXPERIMENTS
from utils.experiments.runner import ExperimentRunner
from utils.explainers.shap_explainer import ShapExplainer
from utils.explainers.permutation_explainer import PermutationExplainer
from utils.experiments.experiment_pipeline import ExperimentPipeline

app = FastAPI()

import time
import shap
import numpy as np


@app.on_event("startup")
def run_experiment_on_startup():

    print("🚀 Starting experiment pipeline...")

    dataset_name = "medical"
    config = EXPERIMENTS[dataset_name]

    # 🔥 możesz tu wrzucać dowolne modele
    models = ["rf", "logreg", "xgb"]

    pipeline = ExperimentPipeline(config, models)
    results = pipeline.run()

    print("\n🔥 FINAL RESULTS:")
    for model_name, data in results.items():
        print(f"\nModel: {model_name}")
        print("Metrics:", data["metrics"])



@app.get("/")
def root():
    return {"message": "Hello from xai-module"}


# rabbitmq_service.consume(
#     queue=XAI_REQUEST_QUEUE,
#     callback=rabbitmq_service.handle_request
# )
from fastapi import FastAPI

from configuration.rabbitmq_queues import XAI_REQUEST_QUEUE
from services import rabbitmq_service

from configuration.experiment_config import EXPERIMENTS
from utils.experiments.runner import ExperimentRunner
from utils.explainers.shap_explainer import ShapExplainer
from utils.explainers.permutation_explainer import PermutationExplainer
from utils.experiments.experiment_pipeline import ExperimentPipeline

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

app = FastAPI()

import time
import shap
import numpy as np


@app.on_event("startup")
def run_experiment_on_startup():
    print("🚀 Starting all experiment pipelines...")

    for experiment_name, config in EXPERIMENTS.items():
        
        print("\n====================================")
        print(f"🚀 Running experiment: {experiment_name}")
        print("====================================")
    
        models = config.get("models")
    
        if models is None:
            if config["task"] == "classification":
                models = ["rf", "logreg", "xgb", "dt"]
            else:
                models = ["linreg", "rf", "xgb"]
    
        pipeline = ExperimentPipeline(config, models)

        try:
            results = pipeline.run()

            print(f"\n✅ Finished experiment: {experiment_name}")

            for model_name, data in results.items():
                print(f"\nModel: {model_name}")
                print("Metrics:", data["metrics"])

        except Exception as e:
            print(f"\n❌ Experiment failed: {experiment_name}")
            print(str(e))

    print("\n🎉 All experiments finished.")



@app.get("/")
def root():
    return {"message": "Hello from xai-module"}


# rabbitmq_service.consume(
#     queue=XAI_REQUEST_QUEUE,
#     callback=rabbitmq_service.handle_request
# )
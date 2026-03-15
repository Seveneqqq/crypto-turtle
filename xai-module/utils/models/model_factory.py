import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor


class ModelFactory:

    @staticmethod
    def create(model_name: str, task: str):

        if task == "classification":

            models = {

                # =========================
                # LOGISTIC REGRESSION
                # =========================
                "logreg": LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",          
                    multi_class="multinomial",
                    n_jobs=-1
                ),

                # =========================
                # RANDOM FOREST
                # =========================
                "rf": RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced"
                ),

                # =========================
                # DECISION TREE
                # =========================
                "dt": DecisionTreeClassifier(
                    random_state=42
                ),

                # =========================
                # MLP
                # =========================
                "mlp": MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                ),

                # =========================
                # XGBOOST (MULTI-CLASS SAFE)
                # =========================
                "xgb": XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    objective="multi:softprob",  
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    n_jobs=-1
                )
            }

        elif task == "regression":

            models = {

                # =========================
                # LINEAR REGRESSION
                # =========================
                "linreg": LinearRegression(),

                # =========================
                # RANDOM FOREST REGRESSOR
                # =========================
                "rf": RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

                # =========================
                # XGBOOST REGRESSOR
                # =========================
                "xgb": XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            }

        else:
            raise ValueError("Task must be classification or regression")

        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")

        return models[model_name]
import time
import pandas as pd

from utils.experiments.runner import ExperimentRunner
from utils.explainers.xai_factory import XaiFactory
from utils.reporting.result_logger import ResultLogger

from utils.evaluation.xai_comparator import XaiComparator
from utils.evaluation.xai_summary import XaiSummary
from utils.evaluation.xai_stability import XaiStability
from utils.evaluation.xai_fidelity import XaiFidelity
from utils.evaluation.xai_local import XaiLocal
from utils.evaluation.xai_debugger import XaiDebugger
from utils.evaluation.xai_ranking import XaiRanking
from utils.evaluation.model_interpretability_analysis import ModelInterpretabilityAnalysis

from utils.reporting.feature_importance_report import FeatureImportanceReport
from utils.reporting.feature_comparison import FeatureComparison

from utils.visualization.xai_heatmap import XaiHeatmap
from utils.visualization.feature_importance_plot import FeatureImportancePlot
from utils.visualization.model_tradeoff_plot import ModelTradeoffPlot

from utils.explanations.human_xai_explainer import HumanXaiExplainer


class ExperimentPipeline:

    def __init__(self, config: dict, model_names: list):
        self.config = config
        self.model_names = model_names

    def run(self):

        results = {}

        logger = ResultLogger(dataset_name=self.config["dataset_name"])

        for model_name in self.model_names:

            print("\n============================")
            print(f"🚀 Running model: {model_name}")
            print("============================")

            start_model = time.time()

            runner = ExperimentRunner(self.config, model_name)

            result = runner.run()

            pipeline = result["pipeline"]
            X_train = result["X_train"]
            X_test = result["X_test"]
            y_test = result["y_test"]
            y_pred = result["y_pred"]

            # ======================
            # XAI METHODS
            # ======================

            xai_results = {}

            for method in self.config.get("xai_methods", ["shap", "permutation"]):

                print(f"🔥 Running {method.upper()}...")

                start = time.time()

                explainer = XaiFactory.create(
                    method,
                    pipeline,
                    X_train,
                    task=self.config["task"]
                )

                # ======================
                # SHAP
                # ======================

                if method == "shap":

                    sample = X_train.sample(
                        min(50, len(X_train)),
                        random_state=42
                    )

                    shap_values = explainer.explain(sample)

                    importance = explainer.get_global_importance(
                        shap_values
                    )

                # ======================
                # PERMUTATION
                # ======================

                elif method == "permutation":

                    perm_result = explainer.explain(
                        X_test.iloc[:300],
                        y_test.iloc[:300]
                    )

                    importance = explainer.get_global_importance(
                        perm_result
                    )

                # ======================
                # LIME
                # ======================

                elif method == "lime":

                    importances = {}

                    samples = X_test.iloc[:min(50, len(X_test))]

                    for i in range(len(samples)):

                        explanation = explainer.explain(
                            samples.iloc[i]
                        )

                        for feature, weight in explanation.items():

                            importances[feature] = (
                                importances.get(feature, 0)
                                + abs(weight)
                            )

                    for k in importances:
                        importances[k] /= len(samples)

                    importance = dict(
                        sorted(
                            importances.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    )

                else:
                    raise ValueError(
                        f"Unknown XAI method: {method}"
                    )

                runtime = time.time() - start

                print(f"⏱ {method} time: {runtime:.2f}s")

                print(f"📊 {method.upper()} top 5 features:")

                for i, (k, v) in enumerate(importance.items()):

                    if i == 5:
                        break

                    print(f"   {k}: {v:.6f}")

                # ======================
                # FEATURE IMPORTANCE REPORT
                # ======================

                importance_df = FeatureImportanceReport.build_table(
                    importance
                )

                importance_path = (
                    f"{logger.base_dir}/"
                    f"{model_name}_{method}_feature_importance.csv"
                )

                importance_df.to_csv(
                    importance_path,
                    index=False
                )

                # ======================
                # FEATURE IMPORTANCE PLOT
                # ======================

                plot_path = (
                    f"{logger.base_dir}/"
                    f"{model_name}_{method}_importance.png"
                )

                FeatureImportancePlot.plot(
                    importance,
                    plot_path
                )

                # ======================
                # STABILITY TEST
                # ======================

                X_test_noisy = XaiStability.add_noise(
                    X_test.iloc[:300]
                )

                if method == "shap":

                    sample_noisy = X_test_noisy.sample(
                        min(50, len(X_test_noisy)),
                        random_state=42
                    )

                    shap_values_noisy = explainer.explain(
                        sample_noisy
                    )

                    importance_noisy = (
                        explainer.get_global_importance(
                            shap_values_noisy
                        )
                    )

                elif method == "permutation":

                    perm_result_noisy = explainer.explain(
                        X_test_noisy.iloc[:300],
                        y_test.iloc[:300]
                    )

                    importance_noisy = (
                        explainer.get_global_importance(
                            perm_result_noisy
                        )
                    )

                elif method == "lime":

                    importance_noisy = importance

                stability = XaiStability.stability_score(
                    importance,
                    importance_noisy
                )

                # ======================
                # FIDELITY
                # ======================

                fidelity = XaiFidelity.feature_removal_test(
                    pipeline,
                    X_test.iloc[:300],
                    y_test.iloc[:300],
                    importance,
                    task=self.config["task"],
                    k=5
                )

                # ======================
                # LOCAL EXPLANATIONS
                # ======================

                local_explanations = XaiLocal.generate_local_explanations(
                    method,
                    explainer,
                    X_test,
                    max_samples=5
                )

                # ======================
                # DEBUGGING ERRORS
                # ======================

                debug_explanations = XaiDebugger.explain_errors(
                    task=self.config["task"],
                    method=method,
                    explainer=explainer,
                    X_test=X_test,
                    y_true=y_test,
                    y_pred=y_pred,
                    max_cases=5
                )

                xai_results[method] = {
                    "global_importance": importance,
                    "stability": float(stability),
                    "runtime": float(runtime),
                    "fidelity": fidelity,
                    "local_explanations": local_explanations,
                    "debug_explanations": debug_explanations
                }

                # ======================
                # 🔵 HUMAN EXPLANATION (DODANE)
                # ======================

                instance = X_test.iloc[0]

                human_explanation = HumanXaiExplainer.generate_full_explanation(
                    instance=instance,
                    prediction=y_pred[0],
                    importance=importance,
                    stability=stability,
                    fidelity=fidelity,
                    comparisons={},
                    method_name=method,
                    task=self.config["task"]
                )

                print("\n🧠 HUMAN EXPLANATION")
                print(human_explanation)

                human_path = (
                    f"{logger.base_dir}/"
                    f"{model_name}_{method}_human_explanation.txt"
                )

                with open(human_path, "w", encoding="utf-8") as f:
                    f.write(human_explanation)

            # ======================
            # FEATURE COMPARISON
            # ======================

            method_importances = {
                m: xai_results[m]["global_importance"]
                for m in xai_results
            }

            comparison_df = FeatureComparison.build(
                method_importances
            )

            comparison_path = (
                f"{logger.base_dir}/"
                f"{model_name}_feature_comparison.csv"
            )

            comparison_df.to_csv(
                comparison_path,
                index=False
            )

            # ======================
            # XAI METHOD COMPARISON
            # ======================

            methods = list(xai_results.keys())

            comparisons = {}

            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):

                    m1 = methods[i]
                    m2 = methods[j]

                    imp1 = xai_results[m1]["global_importance"]
                    imp2 = xai_results[m2]["global_importance"]

                    spearman = XaiComparator.spearman(
                        imp1,
                        imp2
                    )

                    overlap = XaiComparator.top_k_overlap(
                        imp1,
                        imp2
                    )

                    comparisons[f"{m1}_vs_{m2}"] = {
                        "spearman": float(spearman),
                        "top10_overlap": float(overlap)
                    }

            # ======================
            # HEATMAP
            # ======================

            heatmap_path = (
                f"{logger.base_dir}/"
                f"method_agreement_heatmap_{model_name}.png"
            )

            XaiHeatmap.plot(
                comparisons,
                heatmap_path
            )

            # ======================
            # XAI RANKING
            # ======================

            ranking_df = XaiRanking.build_ranking(
                dataset_name=self.config["dataset_name"],
                model_name=model_name,
                xai_results=xai_results,
                comparisons=comparisons
            )

            ranking_path = (
                f"{logger.base_dir}/ranking_{model_name}.csv"
            )

            ranking_df.to_csv(
                ranking_path,
                index=False
            )

            print("\n🏆 XAI Ranking:")
            print(ranking_df)

            print(
                f"⏱ Model total time: "
                f"{time.time() - start_model:.2f}s"
            )

            results[model_name] = {
                "metrics": result["metrics"],
                "xai": xai_results,
                "comparisons": comparisons,
                "ranking": ranking_df.to_dict(
                    orient="records"
                )
            }

            experiment_result = {
                "dataset": self.config["dataset_name"],
                "task": self.config["task"],
                "model": model_name,
                "model_metrics": result["metrics"],
                "xai": xai_results,
                "comparisons": comparisons,
                "ranking": ranking_df.to_dict(
                    orient="records"
                )
            }

            logger.log(experiment_result)

        # ======================
        # SAVE RESULTS
        # ======================

        logger.save()

        summary = XaiSummary.build_summary(
            logger.results
        )

        summary_path = (
            f"{logger.base_dir}/xai_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False
        )

        print("\n📊 XAI SUMMARY")
        print(summary)

        analysis = ModelInterpretabilityAnalysis.build_table(results)

        analysis.to_csv(
            f"{logger.base_dir}/accuracy_vs_interpretability.csv",
            index=False
        )
        
        print("\nAccuracy vs Interpretability:")
        print(analysis)

        # 🔵 DODANE – GENEROWANIE WYKRESU TRADE-OFF
        tradeoff_path = f"{logger.base_dir}/accuracy_vs_interpretability.png"
        ModelTradeoffPlot.plot(analysis, tradeoff_path)

        return results
import time
import pandas as pd
import numpy as np
from utils.evaluation.paired_evaluation import Perturbation, explain_sample, evaluate_stability

from utils.experiments.runner import ExperimentRunner
from utils.explainers.xai_factory import XaiFactory
from utils.reporting.result_logger import ResultLogger

from utils.evaluation.xai_comparator import XaiComparator
from utils.evaluation.xai_summary import XaiSummary
from utils.evaluation.controlled_masking import ControlledMasking
from utils.evaluation.statistical_analysis import noise_comparisons, masking_inference
from utils.reporting.provenance import provenance
from utils.evaluation.xai_local import XaiLocal
from utils.evaluation.xai_debugger import XaiDebugger
from utils.evaluation.xai_ranking import XaiRanking

from utils.reporting.feature_importance_report import FeatureImportanceReport
from utils.reporting.feature_comparison import FeatureComparison

from utils.visualization.xai_heatmap import XaiHeatmap
from utils.visualization.feature_importance_plot import FeatureImportancePlot
from utils.visualization.uncertainty_plot import plot_masking_intervals



class ExperimentPipeline:

    def __init__(self, config: dict, model_names: list):
        self.config = config
        self.model_names = model_names

    def run(self):

        results = {}

        logger = ResultLogger(dataset_name=self.config["dataset_name"])
        self.output_dir = logger.base_dir
        run_provenance = provenance(self.config)

        for model_name in self.model_names:

            print("\n============================")
            print(f" Running model: {model_name}")
            print("============================")

            start_model = time.time()

            runner = ExperimentRunner(self.config, model_name)

            result = runner.run()

            pipeline = result["pipeline"]
            X_train = result["X_train"]
            X_test = result["X_test"]
            y_test = result["y_test"]
            y_pred = result["y_pred"]

            xai_results = {}
            seed = int(self.config.get("xai_seed", 42))
            sample_size = int(self.config.get("xai_sample_size", 50))
            if sample_size < 2:
                raise ValueError("xai_sample_size must be at least 2")
            if len(X_test) < 4:
                raise ValueError("At least four held-out observations are needed")
            midpoint = len(X_test)//2
            explanation_pool, masking_pool = X_test.iloc[:midpoint], X_test.iloc[midpoint:]
            sample = explanation_pool.sample(min(sample_size, len(explanation_pool)), random_state=seed)
            masking_size = min(int(self.config.get("masking_sample_size", 500)), len(masking_pool))
            if masking_size < 2:
                raise ValueError("Masking sample must contain at least two observations")
            mask_sample = (masking_pool.iloc[:masking_size] if self.config.get("split_type") == "time"
                           else masking_pool.sample(masking_size, random_state=seed))
            mask_y = y_test.loc[mask_sample.index]
            masker = ControlledMasking(pipeline, X_train, self.config["task"])
            if not X_test.index.is_unique:
                raise ValueError("Evaluation requires unique observation IDs")
            sample_y = y_test.loc[sample.index]
            noise = Perturbation(X_train,
                level=self.config.get("stability_noise_level", .01),
                discrete_probability=self.config.get("stability_discrete_probability", .01),
                feature_types=self.config.get("stability_feature_types"))
            repeat_seeds = self.config.get("stability_seeds", list(range(101, 121)))

            for method in self.config.get("xai_methods", ["shap", "permutation"]):

                print(f"Running {method.upper()}...")

                start = time.time()

                explainer = XaiFactory.create(
                    method,
                    pipeline,
                    X_train,
                    task=self.config["task"],
                    config=self.config
                )

                importance, _ = explain_sample(method, explainer, sample, sample_y, seed)

                runtime = time.time() - start

                print(f" {method} time: {runtime:.2f}s")
                print(f" {method.upper()} top 5 features:")

                for i, (k, v) in enumerate(importance.items()):

                    if i == 5:
                        break

                    print(f"   {k}: {v:.6f}")

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

                stability_details = evaluate_stability(
                    method, explainer, sample, sample_y, noise, repeat_seeds)
                stability_mean = stability_details["global_summary"]["mean"]
                stability = float("nan") if stability_mean is None else stability_mean
                pd.DataFrame([{
                    "seed": row["seed"], "global_spearman": row["global_spearman"],
                    "local_spearman_mean": row["local_summary"]["mean"],
                    "local_valid_observations": row["local_summary"]["valid_repeats"],
                    "algorithm_seed":row["algorithm_seed"],
                    "control_algorithm_seed":row["control_algorithm_seed"],
                    "data_relative_l1":row["global_relative_l1"],
                    "randomness_spearman":row["randomness_control"]["global_spearman"],
                    "randomness_relative_l1":row["randomness_control"]["global_relative_l1"],
                    "changed_cells": sum(row["changed_rows_per_column"].values())
                } for row in stability_details["repeats"]]).to_csv(
                    f"{logger.base_dir}/{model_name}_{method}_stability_repeats.csv", index=False)

                fidelity = masker.evaluate(mask_sample, mask_y, importance,
                    k=int(self.config.get("masking_k", 5)),
                    seeds=self.config.get("masking_seeds", list(range(501, 521))))
                pd.DataFrame([{k:v for k,v in record.items() if k != "predictions"}
                              for record in fidelity["variants"]]).to_csv(
                    f"{logger.base_dir}/{model_name}_{method}_masking_controls.csv", index=False)

                local_explanations = XaiLocal.generate_local_explanations(
                    method,
                    explainer,
                    sample,
                    max_samples=5
                )

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
                    "stability_details": stability_details,
                    "evaluation_sample_indices": sample.index.tolist(),
                    "output_metadata": explainer.output_metadata() if method != "permutation" else {
                        "output": "estimator_default_score", "note": "PFI is a global score decrease, not a local class explanation."},
                    "background_indices": getattr(explainer, "background_indices", None),
                    "representation": explainer.representation_metadata() if method == "lime" else None,
                    "explanation_algorithm": "permutation_shap" if method == "shap" else method,
                    "ranking_feature_space": "raw_columns" if method == "permutation" else "raw_columns_sum_of_absolute_transformed_weights",
                    "protocol": {"version": "controlled-xai-v4", "seed": seed,
                        "sample_size": len(sample), "stability_seeds": repeat_seeds,
                        "shap_background_size": self.config.get("shap_background_size", 50),
                        "shap_permutations": self.config.get("shap_permutations", 3),
                        "lime_num_samples": self.config.get("lime_num_samples", 5000)},
                    "runtime": float(runtime),
                    "fidelity": fidelity,
                    "local_explanations": local_explanations,
                    "debug_explanations": debug_explanations
                }

                human_explanation = (
                    "Ranking globalny na wspólnej próbie testowej. "
                    "Nie jest objaśnieniem pojedynczej decyzji.\n"
                    f"Metoda: {method}; średnia stabilność rankingu: {stability}.\n"
                    "Zmienność powtórzeń i zakres pomiaru zapisano w stability_details.\n"
                    "Wynik nie stanowi dowodu przewagi metody ani potwierdzenia wierności.\n"
                )

                human_path = (
                    f"{logger.base_dir}/"
                    f"{model_name}_{method}_human_explanation.txt"
                )

                with open(human_path, "w", encoding="utf-8") as f:
                    f.write(human_explanation)

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
                        "scope": "raw_feature_rank_agreement_not_equal_attribution_values",
                        "same_local_output": m1 != "permutation" and m2 != "permutation",
                        "top10_overlap": float(overlap)
                    }

            heatmap_path = (
                f"{logger.base_dir}/"
                f"method_agreement_heatmap_{model_name}.png"
            )

            XaiHeatmap.plot(
                comparisons,
                heatmap_path
            )

            randomness_results = {method:{"stability_details":{"repeats":[
                {"seed":r["seed"],"global_spearman":r["randomness_control"]["global_spearman"]}
                for r in value["stability_details"]["repeats"]]}} for method,value in xai_results.items()}
            randomness_statistics = noise_comparisons(randomness_results, seed=seed,
                resamples=int(self.config.get("permutation_resamples",4999)))
            for row in randomness_statistics:
                row.update(metric="algorithm_randomness_spearman",unit="independent_algorithm_seed_pair",
                    family="algorithm_randomness_pairs_within_dataset_model",
                    scope="fixed_data_model_and_background; algorithm_seed_changes_only")
            pd.DataFrame(randomness_statistics).to_csv(
                f"{logger.base_dir}/{model_name}_randomness_statistics.csv",index=False)
            statistical_results = {
                "algorithm_randomness":randomness_statistics,
                "stability": noise_comparisons(xai_results, seed=seed,
                    resamples=int(self.config.get("permutation_resamples", 4999))),
                "masking": masking_inference(xai_results,
                    temporal=self.config.get("split_type") == "time",
                    horizon=int(self.config.get("finance_horizon", 1)),
                    resamples=int(self.config.get("bootstrap_resamples", 1999)), seed=seed,
                    block_length=self.config.get("bootstrap_block_length"))}
            pd.DataFrame(statistical_results["stability"]).to_csv(
                f"{logger.base_dir}/{model_name}_stability_statistics.csv", index=False)
            pd.DataFrame(statistical_results["masking"]["comparisons"]).to_csv(
                f"{logger.base_dir}/{model_name}_masking_statistics.csv", index=False)
            pd.DataFrame(statistical_results["masking"]["intervals"]).to_csv(
                f"{logger.base_dir}/{model_name}_masking_intervals.csv", index=False)

            plot_masking_intervals(statistical_results["masking"],
                f"{logger.base_dir}/{model_name}_masking_uncertainty.png")

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

            print("\nXAI Ranking:")
            print(ranking_df)

            print(
                f"Model total time: "
                f"{time.time() - start_model:.2f}s"
            )

            results[model_name] = {
                "metrics": result["metrics"],
                "xai": xai_results,
                "comparisons": comparisons,
                "statistics": statistical_results,
                "provenance": run_provenance,
                "data_preparation": result.get("data_preparation"),
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
                "statistics": statistical_results,
                "provenance": run_provenance,
                "data_preparation": result.get("data_preparation"),
                "ranking": ranking_df.to_dict(
                    orient="records"
                )
            }

            logger.log(experiment_result)
            logger.save()  

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

        print("\nXAI SUMMARY")
        print(summary)

        return results

import time

from utils.experiments.runner import ExperimentRunner
from utils.explainers.xai_factory import XaiFactory
from utils.reporting.result_logger import ResultLogger

from utils.evaluation.xai_comparator import XaiComparator
from utils.evaluation.xai_summary import XaiSummary
from utils.evaluation.xai_stability import XaiStability

from utils.visualization.xai_heatmap import XaiHeatmap

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
    
            # ======================
            # XAI METHODS
            # ======================
    
            xai_results = {}
    
            for method in self.config.get("xai_methods", ["shap", "permutation"]):
            
                print(f"🔥 Running {method.upper()}...")
    
                start = time.time()
    
                explainer = XaiFactory.create(method, pipeline, X_train)
    
                # ======================
                # SHAP
                # ======================
    
                if method == "shap":
                
                    sample = X_train.sample(50, random_state=42)
    
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
                # LIME (aggregated)
                # ======================
    
                elif method == "lime":
                
                    importances = {}
    
                    samples = X_test.iloc[:50]
    
                    for i in range(len(samples)):
                    
                        explanation = explainer.explain(samples.iloc[i])
    
                        for feature, weight in explanation.items():
                        
                            importances[feature] = (
                                importances.get(feature, 0) + abs(weight)
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
                
                    raise ValueError(f"Unknown XAI method: {method}")
    
                print(f"⏱ {method} time: {time.time() - start:.2f}s")
    
                print(f"📊 {method.upper()} top 5 features:")
    
                for i, (k, v) in enumerate(importance.items()):
                
                    if i == 5:
                        break
                    
                    print(f"   {k}: {v:.6f}")
    
                # ======================
                # STABILITY TEST
                # ======================
    
                X_test_noisy = XaiStability.add_noise(X_test.iloc[:300])
    
                if method == "shap":
                
                    sample_noisy = X_test_noisy.sample(50, random_state=42)
    
                    shap_values_noisy = explainer.explain(sample_noisy)
    
                    importance_noisy = explainer.get_global_importance(
                        shap_values_noisy
                    )
    
                elif method == "permutation":
                
                    perm_result_noisy = explainer.explain(
                        X_test_noisy.iloc[:300],
                        y_test.iloc[:300]
                    )
    
                    importance_noisy = explainer.get_global_importance(
                        perm_result_noisy
                    )
    
                elif method == "lime":
                
                    importance_noisy = importance
    
                stability = XaiStability.stability_score(
                    importance,
                    importance_noisy
                )
    
                xai_results[method] = {
                    "global_importance": importance,
                    "stability": stability
                }
    
            # ======================
            # XAI COMPARISON
            # ======================
    
            methods = list(xai_results.keys())
    
            comparisons = {}
    
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                
                    m1 = methods[i]
                    m2 = methods[j]
    
                    imp1 = xai_results[m1]["global_importance"]
                    imp2 = xai_results[m2]["global_importance"]
    
                    spearman = XaiComparator.spearman(imp1, imp2)
    
                    overlap = XaiComparator.top_k_overlap(imp1, imp2)
    
                    comparisons[f"{m1}_vs_{m2}"] = {
                        "spearman": spearman,
                        "top10_overlap": overlap
                    }
    
            # ======================
            # HEATMAP
            # ======================
    
            heatmap_path = f"{logger.base_dir}/method_agreement_heatmap_{model_name}.png"
    
            XaiHeatmap.plot(
                comparisons,
                heatmap_path
            )
    
            print(f"⏱ Model total time: {time.time() - start_model:.2f}s")
    
            # ======================
            # RESULTS
            # ======================
    
            results[model_name] = {
                "metrics": result["metrics"],
                "xai": xai_results,
                "comparisons": comparisons
            }
    
            experiment_result = {
            
                "dataset": self.config["dataset_name"],
                "task": self.config["task"],
                "model": model_name,
    
                "model_metrics": result["metrics"],
    
                "xai": xai_results,
    
                "comparisons": comparisons
            }
    
            logger.log(experiment_result)
    
        # ======================
        # SAVE JSON RESULTS
        # ======================
    
        logger.save()
    
        # ======================
        # BUILD SUMMARY
        # ======================
    
        summary = XaiSummary.build_summary(logger.results)
    
        summary_path = f"{logger.base_dir}/xai_summary.csv"
    
        summary.to_csv(summary_path, index=False)
    
        print("\n📊 XAI SUMMARY")
        print(summary)
    
        return results
    
    
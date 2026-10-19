import pandas as pd


class XaiSummary:
    @staticmethod
    def build_summary(results):
        rows = []

        for exp in results:
            dataset = exp["dataset"]
            model = exp["model"]
            comparisons = exp["comparisons"]
            xai = exp["xai"]

            for method, vals in xai.items():
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "stability": vals.get("stability"),
                    "algorithm_randomness_spearman":vals.get("stability_details",{}).get("randomness_summary",{}).get("mean"),
                    "algorithm_randomness_std":vals.get("stability_details",{}).get("randomness_summary",{}).get("std"),
                    "algorithm_randomness_relative_l1":vals.get("stability_details",{}).get("randomness_relative_l1_summary",{}).get("mean"),
                    "stability_std": vals.get("stability_details", {}).get("global_summary", {}).get("std"),
                    "stability_valid_repeats": vals.get("stability_details", {}).get("global_summary", {}).get("valid_repeats"),
                    "runtime": vals.get("runtime"),
                    "fidelity_drop": vals.get("fidelity", {}).get("fidelity_drop")
                })

            for comp_name, comp_vals in comparisons.items():
                m1, m2 = comp_name.split("_vs_")
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "method": f"{m1}_vs_{m2}",
                    "stability": None,
                    "runtime": None,
                    "fidelity_drop": None,
                    "spearman": comp_vals.get("spearman"),
                    "top10_overlap": comp_vals.get("top10_overlap")
                })

        return pd.DataFrame(rows)

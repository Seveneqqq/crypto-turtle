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
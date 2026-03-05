import pandas as pd


class XaiSummary:

    @staticmethod
    def build_summary(results):

        rows = []

        for exp in results:

            dataset = exp["dataset"]
            model = exp["model"]

            comparisons = exp["comparisons"]

            shap_perm = comparisons["shap_vs_permutation"]["spearman"]
            shap_lime = comparisons["shap_vs_lime"]["spearman"]
            perm_lime = comparisons["permutation_vs_lime"]["spearman"]

            overlap_sp = comparisons["shap_vs_permutation"]["top10_overlap"]
            overlap_sl = comparisons["shap_vs_lime"]["top10_overlap"]
            overlap_pl = comparisons["permutation_vs_lime"]["top10_overlap"]

            rows.append({
                "dataset": dataset,
                "model": model,

                "spearman_shap_perm": shap_perm,
                "spearman_shap_lime": shap_lime,
                "spearman_perm_lime": perm_lime,

                "overlap_shap_perm": overlap_sp,
                "overlap_shap_lime": overlap_sl,
                "overlap_perm_lime": overlap_pl
            })

        df = pd.DataFrame(rows)

        return df
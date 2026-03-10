import pandas as pd


class XaiRanking:
    @staticmethod
    def build_ranking(dataset_name, model_name, xai_results, comparisons):
        rows = []

        agreement_scores = {}

        for comp_name, vals in comparisons.items():
            m1, m2 = comp_name.split("_vs_")
            sp = vals.get("spearman", 0.0)

            agreement_scores.setdefault(m1, []).append(sp)
            agreement_scores.setdefault(m2, []).append(sp)

        for method, vals in xai_results.items():
            stability = vals.get("stability", 0.0)
            fidelity = vals.get("fidelity", {}).get("fidelity_drop", 0.0)
            runtime = vals.get("runtime", 0.0)

            agreement = 0.0
            if method in agreement_scores and len(agreement_scores[method]) > 0:
                agreement = sum(agreement_scores[method]) / len(agreement_scores[method])

            # im większy runtime, tym gorzej
            speed_score = 1 / (1 + runtime)

            # dla fidelity: większy drop = bardziej explanation wpływa na model
            final_score = (
                0.35 * stability +
                0.35 * fidelity +
                0.20 * agreement +
                0.10 * speed_score
            )

            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "method": method,
                "stability": float(stability),
                "fidelity_drop": float(fidelity),
                "agreement_mean": float(agreement),
                "runtime": float(runtime),
                "speed_score": float(speed_score),
                "final_score": float(final_score)
            })

        df = pd.DataFrame(rows).sort_values("final_score", ascending=False)
        return df
class HumanXaiExplainer:

    @staticmethod
    def _describe_prediction(prediction, task):

        if task == "classification":
            return f"The model predicted class '{prediction}'."

        else:
            return f"The model predicted value {prediction:.3f}."


    @staticmethod
    def _describe_top_features(top_features):

        text = "The most influential variables for this decision were:\n"

        for feature, value in top_features[:5]:

            text += f" - {feature} (importance: {value:.4f})\n"

        return text


    @staticmethod
    def _describe_feature_values(instance, top_features):

        text = "\nValues of the most important features:\n"

        for feature, _ in top_features[:5]:

            raw_name = feature.split("__")[-1]

            if raw_name in instance.index:

                val = instance[raw_name]

                text += f" - {raw_name}: {val}\n"

        return text


    @staticmethod
    def _describe_model_reliability(stability, fidelity):

        text = "\nModel explanation reliability:\n"

        if stability > 0.9:
            text += "- Explanations are highly stable.\n"

        elif stability > 0.7:
            text += "- Explanations are moderately stable.\n"

        else:
            text += "- Explanations are unstable.\n"

        drop = fidelity.get("fidelity_drop", 0)

        if drop > 0.15:
            text += "- Important features strongly affect model predictions.\n"

        elif drop > 0.05:
            text += "- Important features moderately affect predictions.\n"

        else:
            text += "- Feature importance has limited impact on predictions.\n"

        return text


    @staticmethod
    def _describe_method_comparison(method_name, comparisons):

        text = "\nComparison with other XAI methods:\n"

        for key, val in comparisons.items():

            if method_name in key:

                spearman = val.get("spearman")

                other = key.replace(method_name, "").replace("_vs_", "")

                text += (
                    f"- Agreement with {other}: "
                    f"Spearman correlation = {spearman:.2f}\n"
                )

        return text


    @staticmethod
    def generate_full_explanation(
        instance,
        prediction,
        importance,
        stability,
        fidelity,
        comparisons,
        method_name,
        task="classification"
    ):

        explanation = ""

        explanation += HumanXaiExplainer._describe_prediction(
            prediction,
            task
        )

        top_features = list(importance.items())[:5]

        explanation += "\n\n"
        explanation += HumanXaiExplainer._describe_top_features(
            top_features
        )

        explanation += HumanXaiExplainer._describe_feature_values(
            instance,
            top_features
        )

        explanation += HumanXaiExplainer._describe_model_reliability(
            stability,
            fidelity
        )

        explanation += HumanXaiExplainer._describe_method_comparison(
            method_name,
            comparisons
        )

        return explanation
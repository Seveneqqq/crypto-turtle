"""Output-aligned diagnostics. These are conditional interventions, not proof of fidelity."""
import copy
import numpy as np
from sklearn.inspection import permutation_importance
from utils.evaluation.controlled_masking import ControlledMasking
from utils.evaluation.xai_debugger import XaiDebugger


class ScalarOutput:
    def __init__(self, pipeline, task, class_index=None):
        self.pipeline, self.task = pipeline, task
        self.named_steps = pipeline.named_steps
        self.class_index = None
        self.class_label = None
        if task == 'classification':
            classes = pipeline.named_steps['model'].classes_
            self.class_index = len(classes)-1 if class_index is None else int(class_index)
            if not 0 <= self.class_index < len(classes):
                raise ValueError('Invalid explained class index')
            self.class_label = classes[self.class_index].item() if hasattr(classes[self.class_index], 'item') else classes[self.class_index]
        self.cached_X = self.cached_prediction = None

    def predict(self, X):
        if self.cached_X is not None and X.equals(self.cached_X):
            return self.cached_prediction.copy()
        values = (self.pipeline.predict_proba(X)[:, self.class_index]
                  if self.task == 'classification' else self.pipeline.predict(X))
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite scalar model output')
        return values

    def cache_baseline(self, X):
        self.cached_prediction = self.predict(X).copy()
        self.cached_X = X.copy()

    def target(self, y):
        return (y == self.class_label).astype(float) if self.task == 'classification' else y.astype(float)

    def metadata(self):
        return {'output': 'probability' if self.task == 'classification' else 'prediction',
                'class_index': self.class_index, 'class_label': self.class_label}


class OutputPermutation:
    """PFI loss increase, using the SAME scalar output as SHAP and LIME."""
    def __init__(self, output, repeats=3):
        self.output, self.repeats = output, repeats

    def explain(self, X, y, random_state=42):
        def scoring(estimator, data, truth):
            target = self.output.target(truth)
            return -float(np.sqrt(np.mean((np.asarray(target)-self.output.predict(data))**2)))
        result = permutation_importance(self.output.pipeline, X, y,
            scoring=scoring, n_repeats=self.repeats, random_state=random_state, n_jobs=1)
        return dict(zip(X.columns, map(float, result.importances_mean)))

    def get_global_importance(self, result):
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def masking_reports(output, training, sample, truth, importance, k=5, seeds=range(501,521)):
    output.cache_baseline(sample) if output.cached_X is None else None
    performance = ControlledMasking(output, training, 'regression').evaluate(
        sample, output.target(truth), importance, k=k, seeds=seeds)
    performance.update(measurement='root_brier_degradation' if output.task == 'classification' else 'rmse_degradation',
        output_metadata=output.metadata(), observed_labels=truth.tolist(),
        interpretation='Predictive loss change for one declared output; not explanation fidelity.')
    sensitivity = copy.deepcopy(performance)
    baseline = np.asarray(performance['baseline_predictions'])
    sensitivity['y_true'] = baseline.tolist()  
    sensitivity['baseline_score'] = 0.
    for variant in sensitivity['variants']:
        variant['score'] = variant['drop'] = float(np.sqrt(np.mean((np.asarray(variant['predictions'])-baseline)**2)))
    top, bottom = sensitivity['variants'][:2]
    random = np.mean([v['drop'] for v in sensitivity['variants'][2:]])
    sensitivity.update(measurement='rms_change_of_explained_output',
        reference_target='unmodified_model_output_not_observed_ground_truth',
        perturbed_score=top['score'], fidelity_drop=top['drop'],
        top_minus_bottom=top['drop']-bottom['drop'], top_minus_random=float(top['drop']-random),
        interpretation='Sensitivity of the declared model output to replacement. Not a complete fidelity measure; correlated features may yield unrealistic inputs.')
    return performance, sensitivity


def error_explanations(method, explainer, X, truth, predictions, task, max_cases=5):
    truth_array, predictions = np.asarray(truth), np.asarray(predictions)
    positions = np.flatnonzero(XaiDebugger.find_errors(task, truth_array, predictions))
    if task == 'regression':
        positions = positions[np.argsort(-abs(truth_array[positions]-predictions[positions]), kind='stable')]
    reports = []
    original = explainer.class_index, explainer.class_label
    try:
        for pos in positions[:max_cases]:
            if task == 'classification':
                explainer.class_index = int(np.flatnonzero(explainer.model.classes_ == predictions[pos])[0])
                label = explainer.model.classes_[explainer.class_index]
                explainer.class_label = label.item() if hasattr(label,'item') else label
            row = XaiDebugger.explain_errors(task, method, explainer, X.iloc[[pos]],
                truth.iloc[[pos]], predictions[[pos]], max_cases=1)[0]
            row['explained_target'] = 'predicted_class' if task == 'classification' else 'regression_prediction'
            if method == 'lime':
                row['local_surrogate_fit_score'] = explainer.last_fit_score
            reports.append(row)
    finally:
        explainer.class_index, explainer.class_label = original
    return reports

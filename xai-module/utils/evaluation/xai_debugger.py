import numpy as np
from utils.evaluation.xai_local import XaiLocal

class XaiDebugger:
    @staticmethod
    def find_errors(task, y_true, y_pred):
        truth, predicted = np.asarray(y_true), np.asarray(y_pred)
        if task == 'classification':
            return truth != predicted
        error = np.abs(truth-predicted)
        return (error > 0) & (error >= np.quantile(error, .90))

    @staticmethod
    def explain_errors(task, method, explainer, X_test, y_true, y_pred, max_cases=5):
        truth, predicted = np.asarray(y_true), np.asarray(y_pred)
        if len(truth) != len(X_test) or len(predicted) != len(X_test):
            raise ValueError('Error-report length mismatch')
        if hasattr(y_true, 'index') and not X_test.index.equals(y_true.index):
            raise ValueError('Error-report indices are misaligned')
        positions = np.flatnonzero(XaiDebugger.find_errors(task, truth, predicted))
        if task == 'regression':
            positions = positions[np.argsort(-np.abs(truth[positions]-predicted[positions]), kind='stable')]
        positions = positions[:max_cases]
        if not len(positions):
            return []
        subset = X_test.iloc[positions]
        local = XaiLocal.generate_local_explanations(method, explainer, subset, max_samples=max_cases) if method != 'permutation' else []
        reports = []
        for i, position in enumerate(positions):
            report = dict(local[i]) if method != 'permutation' else {
                'sample_index':int(subset.index[i]), 'top_features':[],
                'note':'PFI has no local explanation for this error.'}
            actual, prediction = truth[position], predicted[position]
            report.update(y_true=actual.item() if hasattr(actual,'item') else actual,
                          prediction=prediction.item() if hasattr(prediction,'item') else prediction)
            if task == 'regression':
                report['signed_error_prediction_minus_truth'] = float(predicted[position]-truth[position])
                report['absolute_error'] = float(abs(predicted[position]-truth[position]))
            else:
                report['incorrect'] = True
            reports.append(report)
        return reports

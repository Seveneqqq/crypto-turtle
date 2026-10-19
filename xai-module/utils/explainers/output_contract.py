import numpy as np

class OutputContract:
    """One fixed output shared by both local explanation methods."""
    def __init__(self, pipeline, task, class_index=None):
        self.model = pipeline.named_steps['model']
        self.preprocessor = pipeline.named_steps['preprocessing']
        self.task = task
        self.class_index = None
        self.class_label = None
        if task == 'classification':
            classes = self.model.classes_
            self.class_index = len(classes)-1 if class_index is None else int(class_index)
            if not 0 <= self.class_index < len(classes):
                raise ValueError('xai_class_index outside model.classes_')
            label = classes[self.class_index]
            self.class_label = label.item() if hasattr(label, 'item') else label
        elif task != 'regression':
            raise ValueError('Unknown task')

    def transform(self, X):
        values = self.preprocessor.transform(X)
        return np.asarray(values.toarray() if hasattr(values, 'toarray') else values, dtype=float)

    def predict_output(self, values):
        if self.task == 'classification':
            return self.model.predict_proba(values)[:, self.class_index]
        return self.model.predict(values)

    def output_metadata(self):
        return {'output': 'probability' if self.task == 'classification' else 'prediction',
                'class_index': self.class_index, 'class_label': self.class_label}

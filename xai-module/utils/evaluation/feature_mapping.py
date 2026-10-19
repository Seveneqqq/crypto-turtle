"""Exact provenance for the column-wise preprocessing used in this project."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def transformed_to_raw(preprocessor):
    names = list(preprocessor.get_feature_names_out())
    mapping = {}
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        if not all(isinstance(col, str) for col in columns):
            raise ValueError("Feature mapping requires explicit named input columns")
        final = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer
        if isinstance(final, OneHotEncoder):
            if final.min_frequency is not None or final.max_categories is not None:
                raise ValueError("Grouped one-hot categories require explicit provenance")
            counts = [len(c) for c in final.categories_]
            if final.drop_idx_ is not None:
                counts = [n - int(drop is not None) for n, drop in zip(counts, final.drop_idx_)]
            raw = [col for col, count in zip(columns, counts) for _ in range(count)]
        else:
            output = list(transformer.get_feature_names_out(columns))
            if output != list(columns):
                raise ValueError(f"No one-to-one provenance for transformer {name}")
            raw = list(columns)
        output_names = names[preprocessor.output_indices_[name]]
        if len(output_names) != len(raw):
            raise ValueError(f"Feature provenance size mismatch for {name}")
        for output_name, raw_name in zip(output_names, raw):
            if output_name in mapping:
                raise ValueError(f"Ambiguous transformed name: {output_name}")
            mapping[output_name] = raw_name
    return mapping

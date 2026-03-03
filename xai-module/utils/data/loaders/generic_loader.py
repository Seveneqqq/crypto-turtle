import pandas as pd

class GenericDataLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        df = pd.read_csv(
            self.config["data_path"],
            encoding="utf-8",
            encoding_errors="ignore"
        )
        return df
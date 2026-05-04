import joblib
import mlflow.pyfunc
import numpy as np


class WineQualityWrapper(mlflow.pyfunc.PythonModel):
    """Same wrapper as Chapter 15."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        raw = self.model.predict(model_input)
        clipped = np.clip(raw, 3.0, 9.0)
        return np.round(clipped, 1).tolist()

import joblib
import mlflow.pyfunc
import numpy as np


class WineQualityWrapper(mlflow.pyfunc.PythonModel):
    """Custom pyfunc wrapper for the wine-quality ElasticNet model.

    Adds two business rules around model.predict():
    - clip raw predictions to the valid quality range [3, 9];
    - round to one decimal so consumers always get a clean number.
    """

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        raw = self.model.predict(model_input)
        clipped = np.clip(raw, 3.0, 9.0)
        return np.round(clipped, 1).tolist()

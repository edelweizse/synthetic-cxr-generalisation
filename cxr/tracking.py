"""MLflow tracking: local SQLite store (mlflow.db) with artifacts under mlruns/.

Browse with:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (from the project root)
"""

import mlflow

from . import PROJECT_ROOT

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT = "cxr-pneumonia"


def setup(experiment: str = EXPERIMENT) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested config -> flat {'a.b': value} for mlflow params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for i, item in enumerate(v):
                out.update(flatten(item, f"{key}.{i}."))
        else:
            out[key] = v
    return out

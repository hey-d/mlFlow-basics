import mlflow
import optuna
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sklearn

path = Path(__file__).parent.parent/"mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{path}")
mlflow.set_experiment("hyperParameters tuning")

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        rf__max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_n_estimators= trial.suggest_int("rf_n_estimators", 50, 300, step=10)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 1.0)
        
        params = {
            "max_depth": rf__max_depth,
            "n_estimators": rf_n_estimators,
            "max_features": rf_max_features,
        }
        
        mlflow.log_params(params)
        regressor_obj = RandomForestRegressor(**params)
        regressor_obj.fit(X_train, y_train  )
        
        y_pred = regressor_obj.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        
        mlflow.log_metric("error", error)
        
        mlflow.sklearn.log_model(regressor_obj, name="model")
        trial.set_user_attr("run_id", child_run.info.run_id)
        return error
        
with mlflow.start_run(run_name="study") as run:
    n_trials =30
    mlflow.log_param("n_trials", n_trials)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("best_error", study.best_value)
    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id)
import os
from pathlib import Path
import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

db_path = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}") 
mlflow.set_experiment("MLflow Quickstart")

wine = load_wine()
X= wine.data
y = wine.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


n_estimators=10
max_depth = [20, 50, 75, 100]
for depth in max_depth:
    
    with mlflow.start_run():
        mlflow.sklearn.autolog()
        
        rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, random_state=42)
        model = rf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            "accuracy": accuracy,
        }
        
        print(accuracy)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(7,5))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names
        )
        
        plt.title("wine confusion matrix", fontsize=14, pad=15)
        plt.xlabel("predicted category", fontsize=12, labelpad=10)
        plt.ylabel("actual category", fontsize=12, labelpad=10)
        
        plt.tight_layout()
        plt.savefig("confusionMatrix.jpg")
        print("seaborn confusion matrix saved")
        
        
    
    
    
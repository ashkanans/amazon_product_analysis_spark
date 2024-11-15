import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


class Visualizer:
    @staticmethod
    def plot_roc_curve(model, test_df):
        """Plot the ROC curve for the given model on the test set."""
        predictions = model.transform(test_df)
        results = predictions.select("label", "probability").toPandas()
        results["probability"] = results["probability"].apply(lambda x: x[1])

        fpr, tpr, _ = roc_curve(results["label"], results["probability"])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label="ROC Curve")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show(block=True)

    @staticmethod
    def plot_feature_importances(model, feature_names):
        """Plot feature importances for the Random Forest model."""
        importances = model.featureImportances.toArray()  # Get feature importances
        indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.show(block=True)

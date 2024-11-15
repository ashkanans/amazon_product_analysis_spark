from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


class RandomForestModel:
    @staticmethod
    def train(train_df):
        """Train a Random Forest Classifier model."""
        rf = RandomForestClassifier(featuresCol="features", labelCol="label")
        rf_model = rf.fit(train_df)
        print("Random Forest model trained.")
        return rf_model

    @staticmethod
    def tune(train_df):
        """Tune Random Forest model with hyperparameter grid."""
        rf = RandomForestClassifier(featuresCol="features", labelCol="label")
        paramGrid = (ParamGridBuilder()
                     .addGrid(rf.numTrees, [20, 50, 100])
                     .addGrid(rf.maxDepth, [5, 10, 15])
                     .build())
        print("Random Forest hyperparameter grid defined.")
        return rf, paramGrid

    @staticmethod
    def cross_validate(train_df, rf, paramGrid):
        """Perform cross-validation for Random Forest and return the best model."""
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cv_model = crossval.fit(train_df)
        print("Cross-validation completed. Best model selected based on AUC.")
        return cv_model.bestModel

    @staticmethod
    def save_model(model, path="random_forest_model"):
        """Save the Random Forest model to the specified path."""
        model.write().overwrite().save(path)
        print(f"Random Forest model saved at: {path}")

    @staticmethod
    def load_model(path="random_forest_model"):
        """Load the Random Forest model from the specified path."""
        from pyspark.ml.classification import RandomForestClassificationModel
        model = RandomForestClassificationModel.load(path)
        print(f"Random Forest model loaded from: {path}")
        return model

    @staticmethod
    def predict(best_rf_model, test_df):
        """Use the best Random Forest model to make predictions on the test set."""
        predictions = best_rf_model.transform(test_df)
        print("Predictions made with Random Forest model.")
        return predictions

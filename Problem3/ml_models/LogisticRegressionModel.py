from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


class LogisticRegressionModel:
    @staticmethod
    def train(train_df):
        """Train a Logistic Regression model."""
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        lr_model = lr.fit(train_df)
        print("Logistic Regression model trained.")
        return lr_model

    @staticmethod
    def tune(train_df):
        """Tune Logistic Regression model with hyperparameter grid."""
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        lr.write().overwrite().save("models/logistic_regression_model")
        paramGrid = (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                     .build())
        print("Logistic Regression hyperparameter grid defined.")
        return lr, paramGrid

    @staticmethod
    def cross_validate(train_df, lr, paramGrid):
        """Perform cross-validation for Logistic Regression and return the best model."""
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        crossval = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cv_model = crossval.fit(train_df)
        print("Cross-validation completed. Best model selected based on AUC.")
        return cv_model.bestModel

    @staticmethod
    def save_model(model: LogisticRegression, path="logistic_regression_model"):
        """Save the Logistic Regression model to the specified path."""
        model.write().overwrite().save(path)
        print(f"Logistic Regression model saved at: {path}")

    @staticmethod
    def load_model(path="logistic_regression_model"):
        """Load the Logistic Regression model from the specified path."""
        from pyspark.ml.classification import LogisticRegressionModel as SparkLRModel
        model = SparkLRModel.load(path)
        print(f"Logistic Regression model loaded from: {path}")
        return model

    @staticmethod
    def predict(best_lr_model, test_df):
        """Use the best Logistic Regression model to make predictions on the test set."""
        predictions = best_lr_model.transform(test_df)
        print("Predictions made with Logistic Regression model.")
        return predictions

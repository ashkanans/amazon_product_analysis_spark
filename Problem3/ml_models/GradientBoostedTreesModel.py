from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


class GradientBoostedTreesModel:
    @staticmethod
    def train(train_df):
        """Train a Gradient-Boosted Trees Classifier."""
        gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)
        gbt_model = gbt.fit(train_df)
        print("Gradient-Boosted Trees model trained.")
        return gbt_model

    @staticmethod
    def tune(train_df):
        """Tune Gradient-Boosted Trees model with hyperparameter grid."""
        gbt = GBTClassifier(featuresCol="features", labelCol="label")
        paramGrid = (ParamGridBuilder()
                     .addGrid(gbt.maxDepth, [5, 10])
                     .addGrid(gbt.maxIter, [10, 20])
                     .build())
        print("Gradient-Boosted Trees hyperparameter grid defined.")
        return gbt, paramGrid

    @staticmethod
    def cross_validate(train_df, gbt, paramGrid):
        """Perform cross-validation for Gradient-Boosted Trees and return the best model."""
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        crossval = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cv_model = crossval.fit(train_df)
        print("Cross-validation completed. Best model selected based on AUC.")
        return cv_model.bestModel

    @staticmethod
    def predict(best_gbt_model, test_df):
        """Use the best Gradient-Boosted Trees model to make predictions on the test set."""
        predictions = best_gbt_model.transform(test_df)
        print("Predictions made with Gradient-Boosted Trees model.")
        return predictions

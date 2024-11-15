from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


class ModelEvaluator:
    @staticmethod
    def evaluate(predictions):
        """Evaluate the model's performance on the test set and print key metrics."""
        binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        auc = binary_evaluator.evaluate(predictions)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        print(f"Model Evaluation Metrics:")
        print(f"AUC: {auc}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")

        confusion_matrix = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
        print("Confusion Matrix:")
        confusion_matrix.show()

import argparse
import os

from pyspark.sql import SparkSession

from Problem3.analysis.FlightDataAnalyzer import FlightDataAnalyzer
from Problem3.data_preparation.FlightDataLoader import FlightDataLoader
from Problem3.evaluation.ModelEvaluator import ModelEvaluator
from Problem3.evaluation.Visualizer import Visualizer
from Problem3.ml_models.LogisticRegressionModel import LogisticRegressionModel
from Problem3.ml_models.RandomForestModel import RandomForestModel


def main(actions, data_path):
    relative_temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(relative_temp_dir, exist_ok=True)

    spark = (SparkSession.builder.
             appName("FlightDelayPrediction")
             .config("spark.local.dir", relative_temp_dir)
             .config("spark.executor.memory", "12g")
             .config("spark.driver.memory", "8g")
             .config("spark.memory.fraction", "0.8")
             .getOrCreate())

    loader = FlightDataLoader()
    df = None

    for action in actions:
        print(f"Executing action: {action}")

        if action == "download":
            loader.download_data()

        elif action == "load":
            df = loader.load_data(spark)

        elif action == "check_missing":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.check_missing_values()
            else:
                print("Data not loaded. Please load data before checking for missing values.")

        elif action == "handle_missing_values":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
            else:
                print("Data not loaded. Please load data before checking for missing values.")

        elif action == "train_evaluate_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
                analyzer.feature_engineering()
                analyzer.prepare_binary_label()
                train, test = analyzer.split_data()
                lr, paramGrid = LogisticRegressionModel.tune(train)
                best_lr_model = LogisticRegressionModel.cross_validate(train, lr, paramGrid)
                # LogisticRegressionModel.save_model(best_lr_model, path="models/logistic_regression") # saving does not work on windows
                predictions = LogisticRegressionModel.predict(best_lr_model, test)
                ModelEvaluator.evaluate(predictions)
                Visualizer.plot_roc_curve(best_lr_model, test)
            else:
                print("Data not loaded. Please load data before checking for missing values.")

        elif action == "train_evaluate_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
                analyzer.feature_engineering()
                analyzer.prepare_binary_label()
                train, test = analyzer.split_data()
                rf, paramGrid = RandomForestModel.tune(train)
                best_rf_model = RandomForestModel.cross_validate(train, rf, paramGrid)
                # RandomForestModel.save_model(best_rf_model, path="models/random_forest") # saving does not work on windows
                predictions = RandomForestModel.predict(best_rf_model, test)
                ModelEvaluator.evaluate(predictions)
                Visualizer.plot_feature_importances(best_rf_model, analyzer.feature_cols)
            else:
                print("Data not loaded. Please load data before checking for missing values.")

        else:
            print(f"Unknown action: {action}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight delay prediction data handling")

    parser.add_argument("actions", nargs="+", type=str,
                        help="List of actions to perform (e.g., load check_missing handle_missing).")
    parser.add_argument("--data_path", type=str, default="data/raw/flights_sample_3m.csv",
                        help="Path to the data file.")

    args = parser.parse_args()
    main(args.actions, args.data_path)

# main_flight.py

import argparse
import os

from pyspark.sql import SparkSession

from Problem3.analysis.FlightDataAnalyzer import FlightDataAnalyzer
from Problem3.data_preparation.FlightDataLoader import FlightDataLoader


def main(actions, data_path):
    # 'temp' directory relative to the current working directory (for training Random Forest Model which required
    # large amount of memory)
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

        elif action == "basic_eda":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.basic_eda()
            else:
                print("Data not loaded. Please load data before performing EDA.")

        elif action == "enhanced_comprehensive_eda":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.enhanced_comprehensive_eda()
            else:
                print("Data not loaded. Please load data before performing EDA.")

        elif action == "comprehensive_eda":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.comprehensive_eda()
            else:
                print("Data not loaded. Please load data before performing EDA.")

        elif action == "handle_missing":
            if df:
                analyzer = FlightDataAnalyzer(df)
                df = analyzer.handle_missing_values()
                print("Missing values handled.")
            else:
                print("Data not loaded. Please load data before handling missing values.")

        elif action == "feature_engineering":
            if df:
                analyzer = FlightDataAnalyzer(df)
                df = analyzer.feature_engineering()
                print("Feature engineering complete.")
            else:
                print("Data not loaded. Please load data before feature engineering.")

        elif action == "split_data":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                print("Data split into train and test sets.")
            else:
                print("Data not loaded. Please load data before splitting.")

        elif action == "prepare_binary_label":
            if df:
                analyzer = FlightDataAnalyzer(df)
                df = analyzer.prepare_binary_label()
                print("Binary label for delay classification added.")
            else:
                print("Data not loaded. Please load data before preparing binary label.")

        elif action == "train_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
                analyzer.feature_engineering()
                analyzer.prepare_binary_label()
                train, test = analyzer.split_data()
                lr_model = analyzer.train_logistic_regression(train)
                print("Logistic Regression model training completed.")
            else:
                print("Data not loaded. Please load data before training the Logistic Regression model.")

        elif action == "train_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
                analyzer.feature_engineering()
                analyzer.prepare_binary_label()
                train, test = analyzer.split_data()
                rf_model = analyzer.train_random_forest(train)
                print("Random Forest model training completed.")
            else:
                print("Data not loaded. Please load data before training the Random Forest model.")

        elif action == "tune_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                lr, paramGrid = analyzer.tune_logistic_regression(train)
                print("Logistic Regression hyperparameter tuning setup completed.")
            else:
                print("Data not loaded. Please load data before tuning the Logistic Regression model.")

        elif action == "tune_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                rf, paramGrid = analyzer.tune_random_forest(train)
                print("Random Forest hyperparameter tuning setup completed.")
            else:
                print("Data not loaded. Please load data before tuning the Random Forest model.")

        elif action == "cross_validate_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                lr, paramGrid = analyzer.tune_logistic_regression(train)
                best_lr_model = analyzer.cross_validate_model(lr, paramGrid, train)
                print("Cross-validation for Logistic Regression completed. Best model selected.")
            else:
                print("Data not loaded. Please load data before cross-validating the Logistic Regression model.")

        elif action == "cross_validate_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                rf, paramGrid = analyzer.tune_random_forest(train)
                best_rf_model = analyzer.cross_validate_model(rf, paramGrid, train)
                print("Cross-validation for Random Forest completed. Best model selected.")
            else:
                print("Data not loaded. Please load data before cross-validating the Random Forest model.")

        elif action == "predict_with_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                lr, paramGrid = analyzer.tune_logistic_regression(train)
                best_lr_model = analyzer.cross_validate_model(lr, paramGrid, train)
                predictions = analyzer.predict_with_logistic_regression(best_lr_model, test)
                print("Predictions with Logistic Regression model completed.")
            else:
                print("Data not loaded. Please load data before making predictions with the Logistic Regression model.")

        elif action == "predict_with_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                rf, paramGrid = analyzer.tune_random_forest(train)
                best_rf_model = analyzer.cross_validate_model(rf, paramGrid, train)
                predictions = analyzer.predict_with_random_forest(best_rf_model, test)
                print("Predictions with Random Forest model completed.")
            else:
                print("Data not loaded. Please load data before making predictions with the Random Forest model.")

        elif action == "evaluate_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
                analyzer.feature_engineering()
                analyzer.prepare_binary_label()
                train, test = analyzer.split_data(1, 1)
                lr, paramGrid = analyzer.tune_logistic_regression(train)
                best_lr_model = analyzer.cross_validate_model(lr, paramGrid, train)
                predictions = analyzer.predict_with_logistic_regression(best_lr_model, test)
                analyzer.evaluate_model(predictions)
                print("Evaluation of Logistic Regression model completed.")
            else:
                print("Data not loaded. Please load data before evaluating the Logistic Regression model.")

        elif action == "evaluate_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.handle_missing_values()
                analyzer.feature_engineering()
                analyzer.prepare_binary_label()
                train, test = analyzer.split_data()
                rf, paramGrid = analyzer.tune_random_forest(train)
                best_rf_model = analyzer.cross_validate_model(rf, paramGrid, train)
                predictions = analyzer.predict_with_random_forest(best_rf_model, test)
                analyzer.evaluate_model(predictions)
                print("Evaluation of Random Forest model completed.")
            else:
                print("Data not loaded. Please load data before evaluating the Random Forest model.")

        elif action == "plot_roc_curve_logistic_regression":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                lr, paramGrid = analyzer.tune_logistic_regression(train)
                best_lr_model = analyzer.cross_validate_model(lr, paramGrid, train)
                analyzer.plot_roc_curve(best_lr_model, test)
                print("ROC curve plotted for Logistic Regression.")
            else:
                print("Data not loaded. Please load data before plotting ROC curve for Logistic Regression.")

        elif action == "plot_roc_curve_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                rf, paramGrid = analyzer.tune_random_forest(train)
                best_rf_model = analyzer.cross_validate_model(rf, paramGrid, train)
                analyzer.plot_roc_curve(best_rf_model, test)
                print("ROC curve plotted for Random Forest.")
            else:
                print("Data not loaded. Please load data before plotting ROC curve for Random Forest.")

        elif action == "plot_feature_importances_random_forest":
            if df:
                analyzer = FlightDataAnalyzer(df)
                train, test = analyzer.split_data()
                rf, paramGrid = analyzer.tune_random_forest(train)
                best_rf_model = analyzer.cross_validate_model(rf, paramGrid, train)
                analyzer.plot_feature_importances(best_rf_model)
                print("Feature importances plotted for Random Forest.")
            else:
                print("Data not loaded. Please load data before plotting feature importances.")

        elif action == "plot_data_distributions":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.plot_data_distributions()
                print("Data distributions plotted.")
            else:
                print("Data not loaded. Please load data before plotting data distributions.")
        else:
            print(f"Unknown action: {action}")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight delay prediction data handling")

    # Accept a space-separated list of actions
    parser.add_argument("actions", nargs="+", type=str,
                        help="List of actions to perform (e.g., load check_missing basic_eda).")
    parser.add_argument("--data_path", type=str, default="data/raw/flights_sample_3m.csv",
                        help="Path to the data file.")

    args = parser.parse_args()
    main(args.actions, args.data_path)

# main_flight.py

import argparse
from pyspark.sql import SparkSession

from Problem3.analysis.FlightDataAnalyzer import FlightDataAnalyzer
from Problem3.data_preparation.FlightDataLoader import FlightDataLoader


def main(actions, data_path):
    # Initialize Spark session
    spark = SparkSession.builder.appName("FlightDelayPrediction").getOrCreate()

    # Initialize data loader and data frame
    loader = FlightDataLoader()
    df = None

    # Loop through actions and execute each one
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

        elif action == "eda":
            if df:
                analyzer = FlightDataAnalyzer(df)
                analyzer.basic_eda()
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

        else:
            print(f"Unknown action: {action}")

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight delay prediction data handling")

    # Accept a space-separated list of actions
    parser.add_argument("actions", nargs="+", type=str,
                        help="List of actions to perform (e.g., load check_missing eda).")
    parser.add_argument("--data_path", type=str, default="data/raw/flights_sample_3m.csv",
                        help="Path to the data file.")

    args = parser.parse_args()
    main(args.actions, args.data_path)

import os

from kaggle import KaggleApi


class FlightDataLoader:
    def __init__(self, dataset_name="patrickzel/flight-delay-and-cancellation-dataset-2019-2023"):
        self.dataset_name = dataset_name
        self.local_path = "data/raw/flights_sample_3m.csv"

        # Initialize the Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()

    def download_data(self):
        """Downloads the dataset from Kaggle and saves it locally."""
        print("Starting download from Kaggle...")

        # Ensure the data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")

        # Download the dataset into the data folder and unzip it
        self.api.dataset_download_files(self.dataset_name, path="data/raw", unzip=True)
        print(f"Dataset downloaded and extracted to data/")

        return self.local_path

    def load_data(self, spark):
        """Loads the dataset into a PySpark DataFrame."""
        print(f"Loading data from {self.local_path}...")

        if not os.path.exists(self.local_path):
            raise FileNotFoundError(f"{self.local_path} does not exist. Please download the data first.")

        df = spark.read.csv(self.local_path, header=True, inferSchema=True)
        print("Data loaded into DataFrame.")
        return df

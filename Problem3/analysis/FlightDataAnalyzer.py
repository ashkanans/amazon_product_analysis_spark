from pyspark.sql import DataFrame
from pyspark.sql.functions import col, mean, isnan, when, count
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


class FlightDataAnalyzer:
    def __init__(self, df: DataFrame):
        self.df = df

    def check_missing_values(self):
        """Check for null or missing values and display their count."""
        missing_counts = self.df.select(
            [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in self.df.columns]
        )
        missing_counts.show()

    def basic_eda(self):
        """Perform basic EDA including summary statistics and distributions."""
        print("Basic statistics of numerical columns:")
        self.df.describe().show()

        # Plot distributions of delay times
        dep_delay_df = self.df.select("DEP_DELAY").dropna().toPandas()
        arr_delay_df = self.df.select("ARR_DELAY").dropna().toPandas()

        # Plot histogram of departure and arrival delays
        plt.figure(figsize=(10, 5))
        sns.histplot(dep_delay_df["DEP_DELAY"], kde=True, bins=30).set_title("Departure Delay Distribution")
        plt.xlabel("Minutes")
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.histplot(arr_delay_df["ARR_DELAY"], kde=True, bins=30).set_title("Arrival Delay Distribution")
        plt.xlabel("Minutes")
        plt.show()

    def handle_missing_values(self):
        """Handle missing values by imputing or dropping."""
        # Drop rows where crucial columns are null
        cols_to_check = ["DEP_DELAY", "ARR_DELAY", "AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME"]
        self.df = self.df.dropna(subset=cols_to_check)
        print("Dropped rows with missing values in essential columns.")
        return self.df

    def feature_engineering(self):
        """Convert categorical columns and assemble features."""
        categorical_cols = ["AIRLINE", "ORIGIN", "DEST"]
        stages = []

        for col_name in categorical_cols:
            # StringIndexer for categorical features
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_Index")
            encoder = OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=f"{col_name}_Vec")
            stages += [indexer, encoder]

        # Assemble features
        feature_cols = ["CRS_DEP_TIME", "DISTANCE"] + [f"{col}_Vec" for col in categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        stages.append(assembler)

        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(self.df)
        self.df = pipeline_model.transform(self.df)

        # Select only necessary columns for modeling
        self.df = self.df.select("features", "ARR_DELAY")
        return self.df

    def split_data(self):
        """Split data into training and testing sets."""
        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        print("Split data into training and testing sets.")
        return train, test

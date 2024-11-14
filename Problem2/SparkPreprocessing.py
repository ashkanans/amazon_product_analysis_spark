import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark session
spark = SparkSession.builder.appName("AmazonProductPreprocessing").getOrCreate()

# Define NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Define the preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and non-alphanumeric characters
    tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


# Define a UDF (User Defined Function) for Spark
preprocess_udf = udf(preprocess_text, ArrayType(StringType()))


def preprocess_with_pyspark(df):
    """
    Function to preprocess text data using PySpark and NLTK.
    :param df: Pandas DataFrame with a 'Description' column to process
    :return: List of processed descriptions
    """
    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(df[['Description']])

    # Apply preprocessing UDF to the Spark DataFrame
    processed_spark_df = spark_df.withColumn("ProcessedDescription", preprocess_udf(col("Description")))

    # Convert the result back to Pandas for downstream analysis
    processed_df = processed_spark_df.select("ProcessedDescription").toPandas()
    processed_descriptions = processed_df['ProcessedDescription'].tolist()

    return processed_descriptions


# Stop the Spark session after processing
spark.stop()

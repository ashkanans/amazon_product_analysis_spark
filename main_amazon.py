import argparse

import os
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, Row

from Problem1.analysis.LDAAnalyzer import LDAAnalyzer
from Problem1.analysis.WordFrequencyAnalyzer import WordFrequencyAnalyzer
from Problem1.scraping.AmazonScraper import AmazonScraper
from Problem1.search.SearchEngine import SearchEngine
from Problem2.SparkSearchEngine import SparkSearchEngine

venv_python_path = ".venv/Scripts/python.exe"

os.environ["PYSPARK_PYTHON"] = venv_python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python_path

# Configure Spark to use the virtual environment's Python
spark = SparkSession.builder \
    .appName("SparkTest") \
    .master("local[*]") \
    .config("spark.pyspark.python", venv_python_path) \
    .config("spark.pyspark.driver.python", venv_python_path) \
    .getOrCreate()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Amazon product analysis and topic modeling.")
    # Primary flags for tasks
    parser.add_argument("--scrape", action="store_true", help="Scrape Amazon data based on keywords.")
    parser.add_argument("--plot_frequency", action="store_true", help="Plot word, bigram, and trigram frequencies.")
    parser.add_argument("--run_lda", action="store_true", help="Perform LDA topic modeling on the data.")
    parser.add_argument("--run_search", action="store_true", help="Perform search on the indexed data.")

    # Conditional parameters for each flag
    parser.add_argument("--keyword", type=str, default="computer, pc, portatile, laptop",
                        help="Keywords for Amazon search, separated by commas.")
    parser.add_argument("--num_pages", type=int, default=5, help="Number of pages to scrape.")
    parser.add_argument("--path", type=str, default=os.path.join('data', 'raw', 'computer_results_default.tsv'),
                        help="TSV file path for Amazon products.")
    parser.add_argument("--num_topics", type=int, default=5, help="Number of topics for LDA.")
    parser.add_argument("--passes", type=int, default=15, help="Number of passes for LDA training.")
    parser.add_argument("--top_words", type=int, default=10, help="Top words for frequency analysis.")
    parser.add_argument("--top_bigrams", type=int, default=10, help="Top bigrams for frequency analysis.")
    parser.add_argument("--top_trigrams", type=int, default=10, help="Top trigrams for frequency analysis.")
    parser.add_argument("--query", type=str, help="Query for product search.")
    parser.add_argument("--top_k", type=int, default=5, help="Top results to display for search.")
    parser.add_argument("--use_pyspark", action="store_true", help="Enable PySpark for preprocessing.")

    args = parser.parse_args()
    return args


def load_or_scrape_data(args):
    amazon_scraper = AmazonScraper(args.keyword, args.num_pages)
    if args.scrape:
        print("Scraping Amazon products...")
        amazon_scraper.scrape_amazon_products()
        amazon_scraper.save_to_tsv()
        amazon_scraper.load_dataset(amazon_scraper.scraped_results)
    else:
        if os.path.exists(args.path):
            print(f"Loading Amazon data from {args.path}...")
            amazon_scraper.load_dataset(args.path)
        else:
            raise FileNotFoundError(f"Dataset file not found at {args.path}")

    # Preprocess with PySpark if enabled, otherwise use standard processing
    if args.use_pyspark:
        print("Preprocessing data with PySpark...")
        processed_descriptions = preprocess_with_pyspark(amazon_scraper.df)
    else:
        print("Preprocessing data with standard processing...")
        processed_descriptions = amazon_scraper.preprocess_descriptions()

    return processed_descriptions


def perform_frequency_analysis(processed_descriptions, args):
    print("Running word frequency analysis and plotting...")
    frequency_analyzer = WordFrequencyAnalyzer(preprocessed_data=processed_descriptions)
    print("Top words by frequency:", frequency_analyzer.calculate_word_frequency().most_common(args.top_words))
    print("Top bigrams by frequency:", frequency_analyzer.calculate_bigram_frequency().most_common(args.top_bigrams))
    print("Top trigrams by frequency:", frequency_analyzer.calculate_trigram_frequency().most_common(args.top_trigrams))
    frequency_analyzer.plot_top_words(top_n=args.top_words)
    frequency_analyzer.plot_top_bigrams(top_n=args.top_bigrams)
    frequency_analyzer.plot_top_trigrams(top_n=args.top_trigrams)


def perform_lda_analysis(processed_descriptions, args):
    print("Running LDA topic modeling...")
    lda_analyzer = LDAAnalyzer(preprocessed_data=processed_descriptions, num_topics=args.num_topics, passes=args.passes)
    lda_analyzer.prepare_corpus()
    lda_analyzer.run_lda()
    lda_analyzer.display_topics(num_words=args.top_words)
    lda_analyzer.visualize_topics()


def perform_search(processed_descriptions, args):
    print("Indexing and searching Amazon product descriptions...")
    search_engine = SearchEngine(min_score_threshold=0.05)
    documents = {i: ' '.join(desc) for i, desc in enumerate(processed_descriptions)}
    search_engine.index_documents(documents)
    results = search_engine.search(args.query, top_k=args.top_k)
    print("Top search results:")
    for doc_id, score in results:
        print(f"Document ID: {doc_id}, Score: {score}, Description: {' '.join(processed_descriptions[doc_id])}")


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


def preprocess_with_pyspark(df):
    """
    Function to preprocess text data using PySpark and NLTK.
    :param df: Pandas DataFrame with a 'Description' column to process
    :return: List of processed descriptions
    """
    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(df[['Description']])

    preprocess_udf = udf(preprocess_text, ArrayType(StringType()))

    # Apply preprocessing UDF to the Spark DataFrame
    processed_spark_df = spark_df.withColumn("ProcessedDescription", preprocess_udf(col("Description")))

    # Convert the result back to Pandas for downstream analysis
    processed_df = processed_spark_df.select("ProcessedDescription").toPandas()
    processed_descriptions = processed_df['ProcessedDescription'].tolist()

    return processed_descriptions


def main():
    args = parse_arguments()
    processed_descriptions = load_or_scrape_data(args)

    if args.plot_frequency:
        perform_frequency_analysis(processed_descriptions, args)
    if args.run_lda:
        perform_lda_analysis(processed_descriptions, args)
    if args.run_search and args.use_pyspark:
        documents = {i: ' '.join(desc) for i, desc in enumerate(processed_descriptions)}
        print("Using Spark for search and indexing...")

        # Initialize the Spark-based search engine
        search_engine = SparkSearchEngine(min_score_threshold=0.05)

        # Build the inverted index and calculate TF-IDF for Spark-based processing
        search_engine.build_inverted_index(documents)
        search_engine.calculate_tfidf(documents)

        # Perform the query processing
        if not args.query:
            raise ValueError("A search query must be provided with the --run_search flag.")

        results = search_engine.search(args.query, top_k=args.top_k)

        print("Top search results:")
        for result in results:
            doc_id, score = result["doc_id"], result["similarity"]
            print(f"Document ID: {doc_id}, Score: {score}, Description: {' '.join(processed_descriptions[doc_id])}")

    # If non-Spark search is requested
    elif args.run_search and not args.use_pyspark:
        documents = {i: ' '.join(desc) for i, desc in enumerate(processed_descriptions)}
        print("Using non-Spark (original) search engine...")

        # Initialize the original search engine
        search_engine = SearchEngine(min_score_threshold=0.05)

        # Build the TF-IDF matrix for non-Spark processing
        search_engine.index_documents(documents)

        # Perform the query processing
        if not args.query:
            raise ValueError("A search query must be provided with the --run_search flag.")

        results = search_engine.search(args.query, top_k=args.top_k)

        print("Top search results:")
        for doc_id, score in results:
            print(f"Document ID: {doc_id}, Score: {score}, Description: {' '.join(processed_descriptions[doc_id])}")


if __name__ == "__main__":
    main()
    spark.stop()

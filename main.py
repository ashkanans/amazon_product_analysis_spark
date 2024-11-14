import argparse
import os

from Problem1.analysis.LDAAnalyzer import LDAAnalyzer
from Problem1.analysis.WordFrequencyAnalyzer import WordFrequencyAnalyzer
from Problem1.scraping.AmazonScraper import AmazonScraper
from Problem1.search.SearchEngine import SearchEngine
from Problem2.SparkPreprocessing import preprocess_with_pyspark


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


def main():
    args = parse_arguments()
    processed_descriptions = load_or_scrape_data(args)

    if args.plot_frequency:
        perform_frequency_analysis(processed_descriptions, args)
    if args.run_lda:
        perform_lda_analysis(processed_descriptions, args)
    if args.run_search:
        if not args.query:
            raise ValueError("A search query must be provided with the --run_search flag.")
        perform_search(processed_descriptions, args)


if __name__ == "__main__":
    main()

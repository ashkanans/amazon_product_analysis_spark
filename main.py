# import os
#
# from Problem1.AmazonScraper import AmazonScraper
# from Problem1.LDAAnalyzer import LDAAnalyzer
# from Problem1.WordFrequencyAnalyzer import WordFrequencyAnalyzer
#
# keyword = "computer, pc, portatile, laptop"
# num_pages = 10
# amazon_scraper = AmazonScraper(keyword, num_pages)
# # amazon_scraper.scrape_amazon_products()
# # amazon_scraper.save_to_tsv()
#
# amazon_scraper.load_dataset()
#
# # amazon_scraper.analyze_data(amazon_scraper.df)
# processed_descriptions = amazon_scraper.preprocess_descriptions()
#
# # # Instantiate WordFrequencyAnalyzer with preprocessed data
# # frequency_analyzer = WordFrequencyAnalyzer(preprocessed_data=processed_descriptions)
# #
# # # Calculate and display top word frequencies
# # word_freq = frequency_analyzer.calculate_word_frequency()
# # print("Top 10 words by frequency:", word_freq.most_common(10))
# #
# # # Calculate and display top bigram frequencies
# # bigram_freq = frequency_analyzer.calculate_bigram_frequency()
# # print("Top 10 bigrams by frequency:", bigram_freq.most_common(10))
# #
# # # Calculate and display top trigram frequencies
# # trigram_freq = frequency_analyzer.calculate_trigram_frequency()
# # print("Top 10 trigrams by frequency:", trigram_freq.most_common(10))
# #
# # # Plot top words, bigrams, and trigrams
# # frequency_analyzer.plot_top_words(top_n=10)
# # frequency_analyzer.plot_top_bigrams(top_n=10)
# # frequency_analyzer.plot_top_trigrams(top_n=10)
#
# # Instantiate LDAAnalyzer
# lda_analyzer = LDAAnalyzer(preprocessed_data=processed_descriptions, num_topics=5, passes=15)
#
# # Prepare the corpus
# lda_analyzer.prepare_corpus()
#
# # Run LDA
# lda_model = lda_analyzer.run_lda()
#
# # Display the topics
# lda_analyzer.display_topics(num_words=10)
#
# # Visualize the topics
# lda_analyzer.visualize_topics()

import argparse
import os

from Problem1.analysis.LDAAnalyzer import LDAAnalyzer
from Problem1.analysis.WordFrequencyAnalyzer import WordFrequencyAnalyzer
from Problem1.scraping.AmazonScraper import AmazonScraper
from Problem1.search.SearchEngine import SearchEngine


def main():
    scrape = False
    plot_frequency = False
    run_lda = False
    run_search_engine = False

    parser = argparse.ArgumentParser(description="Run Amazon product analysis and topic modeling.")

    # Flags for selective execution
    parser.add_argument(
        "--scrape", action="store_true", default=False,
        help="Include this flag to scrape Amazon data based on the provided keywords."
    )
    parser.add_argument(
        "--plot_frequency", action="store_true",
        help="Include this flag to calculate and plot word, bigram, and trigram frequencies."
    )
    parser.add_argument(
        "--run_lda", action="store_true",
        help="Include this flag to perform LDA topic modeling on the data."
    )
    parser.add_argument(
        "--run_search", action="store_true",
        help="Include this flag to perform search on the indexed data."
    )

    # Parse initial arguments to determine which options are needed
    args, remaining_args = parser.parse_known_args()
    path_default = os.path.join('data', 'raw', 'computer_results_default.tsv')

    # If scraping is requested, ask for keyword and number of pages
    if args.scrape:
        scrape = True
        parser.add_argument(
            "--keyword", type=str, default="computer, pc, portatile, laptop",
            help="Keywords to search on Amazon, separated by commas (e.g., 'computer, pc, laptop')."
        )
        parser.add_argument(
            "--num_pages", type=int, default=5,
            help="Number of pages to scrape from Amazon (e.g., 10)."
        )
        parser.add_argument(
            "--path", type=str, default=path_default,
            help="TSV file path of the Amazon products"
        )
    else:
        parser.add_argument(
            "--keyword", type=str, default="computer",
            help="Keywords that was used to search Amazon"
        )
        parser.add_argument(
            "--path", type=str, default=path_default,
            help="TSV file path of the Amazon products"
        )

    # If LDA is requested, ask for number of topics and passes
    if args.run_lda:
        run_lda = True
        parser.add_argument(
            "--num_topics", type=int, default=5,
            help="Number of topics for LDA topic modeling (e.g., 5)."
        )
        parser.add_argument(
            "--passes", type=int, default=15,
            help="Number of passes for LDA model training (e.g., 15)."
        )

    # If frequency analysis is requested, ask for top N words, bigrams, and trigrams
    if args.plot_frequency:
        plot_frequency = True
        parser.add_argument(
            "--top_words", type=int, default=10,
            help="Number of top words to display in frequency analysis (e.g., 10)."
        )
        parser.add_argument(
            "--top_bigrams", type=int, default=10,
            help="Number of top bigrams to display in frequency analysis (e.g., 10)."
        )
        parser.add_argument(
            "--top_trigrams", type=int, default=10,
            help="Number of top trigrams to display in frequency analysis (e.g., 10)."
        )

    # If search is requested, ask for query and number of results
    if args.run_search:
        run_search_engine = True
        parser.add_argument(
            "--query", type=str, required=True,
            help="Search query for finding relevant Amazon products."
        )
        parser.add_argument(
            "--top_k", type=int, default=5,
            help="Number of top results to display for the search query (e.g., 5)."
        )

    # Parse all arguments including conditionally added ones
    args = parser.parse_args(remaining_args)

    if scrape:
        # Initialize AmazonScraper with keywords and number of pages to scrape
        amazon_scraper = AmazonScraper(args.keyword, args.num_pages)
        print("Scraping Amazon products...")
        amazon_scraper.scrape_amazon_products()
        amazon_scraper.save_to_tsv()
        amazon_scraper.load_dataset(amazon_scraper.scraped_results)
    else:
        # Check if the specified dataset file exists
        if os.path.exists(args.path):
            amazon_scraper = AmazonScraper(args.keyword, None)  # No need for keyword when loading from file
            print(f"Loading Amazon data from {args.path}...")
            amazon_scraper.load_dataset(args.path)
        else:
            raise FileNotFoundError(f"Dataset file not found at {args.path}")

    # Preprocess descriptions
    processed_descriptions = amazon_scraper.preprocess_descriptions()

    # Run Word Frequency Analysis if requested
    if plot_frequency:
        print("Running word frequency analysis and plotting...")
        frequency_analyzer = WordFrequencyAnalyzer(preprocessed_data=processed_descriptions)

        # Display top word frequencies
        word_freq = frequency_analyzer.calculate_word_frequency()
        print("Top words by frequency:", word_freq.most_common(args.top_words))

        # Display top bigram frequencies
        bigram_freq = frequency_analyzer.calculate_bigram_frequency()
        print("Top bigrams by frequency:", bigram_freq.most_common(args.top_bigrams))

        # Display top trigram frequencies
        trigram_freq = frequency_analyzer.calculate_trigram_frequency()
        print("Top trigrams by frequency:", trigram_freq.most_common(args.top_trigrams))

        # Plot top words, bigrams, and trigrams
        frequency_analyzer.plot_top_words(top_n=args.top_words)
        frequency_analyzer.plot_top_bigrams(top_n=args.top_bigrams)
        frequency_analyzer.plot_top_trigrams(top_n=args.top_trigrams)

    # Run LDA Analysis if requested
    if run_lda:
        print("Running LDA topic modeling...")
        lda_analyzer = LDAAnalyzer(preprocessed_data=processed_descriptions, num_topics=args.num_topics,
                                   passes=args.passes)

        # Prepare corpus and run LDA
        lda_analyzer.prepare_corpus()
        lda_model = lda_analyzer.run_lda()

        # Display topics
        lda_analyzer.display_topics(num_words=args.top_words)

        # Visualize topics
        lda_analyzer.visualize_topics()

    # Run Search if requested
    if run_search_engine:
        print("Indexing and searching Amazon product descriptions...")
        search_engine = SearchEngine(min_score_threshold=0.05)

        # Convert processed descriptions to a dictionary with doc_ids
        documents = {i: ' '.join(desc) for i, desc in enumerate(processed_descriptions)}
        search_engine.index_documents(documents)

        # Perform search and display top results
        results = search_engine.search(args.query, top_k=args.top_k)
        print("Top search results:")
        for doc_id, score in results:
            print(f"Document ID: {doc_id}, Score: {score}, Description: {' '.join(processed_descriptions[doc_id])}")


if __name__ == "__main__":
    main()

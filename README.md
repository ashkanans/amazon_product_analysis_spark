
# Product Search and Flight Delay Prediction

---

## Project Description

This project is divided into three main problems:

1. **Amazon Product Search**: Download, preprocess, and analyze product data from Amazon. Build a search engine using an inverted index and cosine similarity.
2. **Spark Implementation of Problem 1**: Implement the same functionality as stated in 1 using Apache Spark.
3. **Flight Delay Prediction**: Use PySpark’s MLlib to build, tune, and evaluate machine learning models predicting flight delays.

---

## Project Structure

```
product_search_and_flight_delay_ml/
├── data/
│   └── raw/
│       ├── computer_results_default.tsv
│       ├── dictionary.html
│       ├── flights_sample_3m.csv
├── files/
│   ├── description/
│   │   └── homework2.pdf
│   ├── eda/
│   │   ├── Average Departure and Arrival Delays by Airline.png
│   │   ├── Distribution of Arrival Delays.png
│   │   ├── Correlation Heatmap for Numerical Features.png
│   │   └── ...
│   ├── report/
│       ├── dm-hw2.pdf
│       └── dm-hw2.tex
├── Problem1/
│   ├── analysis/
│   │   ├── LDAAnalyzer.py
│   │   └── WordFrequencyAnalyzer.py
│   ├── scraping/
│   │   └── AmazonScraper.py
│   ├── search/
│   │   └── SearchEngine.py
│   ├── text_processing/
│       └── TextPreprocessor.py
├── Problem2/
│   ├── SparkLDAAnalyzer.py
│   ├── SparkPreprocessing.py
│   └── SparkSearchEngine.py
├── Problem3/
│   ├── analysis/
│   │   └── FlightDataAnalyzer.py
│   ├── data_preparation/
│   │   └── FlightDataLoader.py
│   ├── evaluation/
│   │   ├── ModelEvaluator.py
│   │   └── Visualizer.py
│   ├── ml_models/
│       ├── LogisticRegressionModel.py
│       └── RandomForestModel.py
├── tests/
│   ├── test_amazon_scraper.py
│   ├── test_lda_analyzer.py
│   ├── test_flight_data_loader.py
│   └── ...
├── LICENSE
├── README.md
├── main_amazon.py
├── main_flight.py
├── requirements.txt
```

---

### Amazon Product Search

This section details the first part of the project: scraping product data from Amazon, preprocessing it, and building a
search engine with an inverted index and cosine similarity.

---

#### Objective

To gather and process data on Amazon products for the keyword **"computer"**. The task involves:

1. Scraping product details such as description, price, prime status, URL, ratings, and reviews.
2. Preprocessing textual descriptions for efficient analysis and search.
3. Building a search engine to allow users to query products based on textual descriptions, ranked by relevance.

---

#### Implementation

##### 1. **Web Scraping**

**AmazonScraper** class handles the extraction of product data using Python libraries:

- **Requests**: To send HTTP requests to Amazon and fetch HTML pages.
- **BeautifulSoup**: For parsing HTML and extracting relevant data.
- **Random Delays**: Introduced using `time.sleep()` to avoid being blocked by Amazon.

**Data Captured**:

- **Description**: A brief overview of the product.
- **Price**: Extracted and converted into a float.
- **Prime Status**: Boolean indicating Prime eligibility.
- **URL**: The product's page link.
- **Rating**: Average star rating.
- **Reviews**: Number of customer reviews.

The scraped data is saved as a `.tsv` file in the `data/raw/` directory.

**Code Reference**: See `AmazonScraper.py` for full implementation.

---

##### 2. **Text Preprocessing**

**TextPreprocessor** class processes product descriptions for analysis:

- **Tokenization**: Splitting text into individual words or tokens.
- **Stopword Removal**: Eliminating common, uninformative words.
- **Lemmatization and Stemming**: Reducing words to their base or root form.
- **Multi-word Term Preservation**: Retaining phrases like "Windows 10" as a single token.

**Purpose**:

- Ensure uniformity in data representation.
- Prepare descriptions for use in the search engine and topic modeling.

**Code Reference**: See `TextPreprocessor.py` for details.

---

##### 3. **Word Frequency Analysis**

**WordFrequencyAnalyzer** calculates and visualizes:

- **Word Frequencies**: Most common terms in product descriptions.
- **Bigrams and Trigrams**: Frequent two- and three-word combinations.

Visualizations include bar charts for top terms to help understand the dataset's content.

**Code Reference**: See `WordFrequencyAnalyzer.py` for implementation.

---

##### 4. **Search Engine**

**SearchEngine** uses TF-IDF (Term Frequency-Inverse Document Frequency) to rank products based on cosine similarity:

- **Inverted Index**: Maps terms to product descriptions containing them.
- **Query Processing**: Allows users to search for products using keywords.
- **Ranking**: Returns results ranked by relevance, considering query-document similarity.

Key Features:

- Supports unigram, bigram, and trigram matches.
- Filters results below a minimum relevance threshold.

**Code Reference**: See `SearchEngine.py`.

---

##### 5. **Topic Modeling**

**LDAAnalyzer** applies Latent Dirichlet Allocation (LDA) to identify themes in product descriptions:

- Groups similar products into topics.
- Displays top words in each topic and visualizes the results.

**Code Reference**: See `LDAAnalyzer.py`.

---

#### Running the Program

**Setup**:

1. Install the dependencies required for the project using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute the main script `main_amazon.py` with your desired options. Below are example commands and the expected
   functionality:

---

**Commands and Examples**

1. **Scraping Data**:
   To scrape Amazon products for specific keywords (e.g., "laptop, pc") and save the results:
   ```bash
   python main_amazon.py --scrape --keyword "laptop, pc" --num_pages 3
   ```
   - Scrapes the first 3 pages for each keyword.
   - Saves the results in a `.tsv` file in `data/raw/`.

2. **Loading Existing Data**:
   To load pre-scraped data from a specific file:
   ```bash
   python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --keyword "laptop, pc"
   ```
   - The program reads the data and preprocesses product descriptions.

3. **Word Frequency Analysis**:
   To analyze and visualize word, bigram, and trigram frequencies:
   ```bash
   python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --plot_frequency --top_words 10 --top_bigrams 10 --top_trigrams 10
   ```
   - Displays and plots the top 10 most frequent words, bigrams, and trigrams.
   #### Top 10 Most Common Words
   ![Top 10 Most Common Words](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Words.png)

   #### Top 10 Most Common Bigrams
   ![Top 10 Most Common Bigrams](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Bigrams.png)

   #### Top 10 Most Common Trigrams
   ![Top 10 Most Common Trigrams](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Trigrams.png)

4. **Topic Modeling (LDA)**:
   To perform LDA topic modeling on the dataset:
   ```bash
   python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --run_lda --num_topics 5 --passes 15
   ```
   - Extracts 5 topics from the dataset with 15 training passes.
   - Displays top words for each topic and visualizes topic distributions.

5. **Search Functionality**:
   To search for products matching specific queries:
   ```bash
   python main_amazon.py --run_search --keyword "laptop, pc" --query "Intel Core i7 SSD HD Ram 16Gb" --top_k 5
   ```
   - Displays the top 5 products most relevant to the query based on cosine similarity.

   Example result:
   ```
   Top search results:
   Document ID: 5, Score: 0.2735, Description: pc fisso computer desktop intel_core_i7 intel hd masterizz wifi interno ...
   Document ID: 173, Score: 0.1756, Description: jumper computer portatile hd display office_365 ...
   ```

6. **Default Behavior**:
   If no `--path` option is specified, the program defaults to loading `data/raw/computer_results_default.tsv`:
   ```bash
   python main_amazon.py --run_search --query "HP Notebook G9 Intel i3" --top_k 5
   ```

---

**Options and Flags**:

- `--scrape`: Scrape Amazon for new data using specified keywords and pages.
- `--path`: Specify a path to a `.tsv` file with pre-scraped data.
- `--plot_frequency`: Generate word, bigram, and trigram frequency visualizations.
- `--run_lda`: Perform LDA topic modeling with configurable topics and passes.
- `--run_search`: Search for products matching a query string.
- `--top_k`: Specify the number of top search results to return.

**Code Reference**: Refer to `main_amazon.py` for the complete CLI implementation.

---

**Example Workflow**:

1. Scrape data:
   ```bash
   python main_amazon.py --scrape --keyword "laptop, pc" --num_pages 3
   ```
2. Visualize word frequency:
   ```bash
   python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --plot_frequency --top_words 10
   ```
   ![Top 10 Most Common Words](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Words.png)

3. Run topic modeling:
   ```bash
   python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --run_lda --num_topics 5
   ```
   ![LDA Topic Modeling](./files/1.%20amazon%20prodcut%20search/LDA%20topic%20modeling.png)

4. Search products:
   ```bash
   python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --run_search --query "Intel Core i7 SSD HD" --top_k 5
   ```

---

#### Deliverables

1. **Scraped Data**: Stored in `data/raw/computer_results_default.tsv`.
2. **Indexed Data**: Preprocessed descriptions indexed for searching.
3. **Analysis and Visualizations**: Word frequencies, bigrams, trigrams, and LDA results.
4. **Search Results**: Ranked products for user-defined queries.
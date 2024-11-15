
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

Here's an improved **Commands and Examples** section that presents each command clearly with sample console output and
relevant images. This structure consolidates information and provides a clear, sequential guide for running each
command.

---

### Commands and Examples

The following commands illustrate various functionalities for the Amazon Product Search task. Each command includes
sample console output and images demonstrating the results.

---

#### 1. Scraping Data

To scrape Amazon products for specific keyword(s) (e.g., "laptop, pc") and save the results to a `.tsv` file:

```bash
python main_amazon.py --scrape --keyword "laptop, pc" --num_pages 3
```

**Console Output**:

```
Scraping Amazon products...
Scraping keyword laptop...
Scraping page 1...
Scraping page 2...
Scraping page 3...
216 products found
Scraping keyword pc...
Scraping page 1...
Scraping page 2...
Scraping page 3...
432 products found
Data saved to file: data/raw/laptop_results_2024-11-15.tsv
```

This command scrapes 3 pages of results for each keyword ("laptop" and "pc") and saves them
in `data/raw/laptop_results_2024-11-15.tsv`.

The naming convetion for this tsv file is: <first_keyword>_<current_date>

---

#### 2. Loading and Preprocessing Existing Data

To load a pre-scraped dataset and preprocess the product descriptions, use the `--path` option:

```bash
python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --keyword "laptop, pc"
```

**Console Output**:

```
Loading Amazon data from data/raw/laptop_results_2024-11-15.tsv...
Data loaded successfully.
Preprocessing data with standard processing...
```

This command loads and preprocesses data from the specified `.tsv` file without additional scraping.


---

#### 3. Word Frequency Analysis

To analyze word, bigram, and trigram frequencies and generate visualizations for the most common terms, run:

```bash
python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --plot_frequency --top_words 10 --top_bigrams 10 --top_trigrams 10
```

**Console Output**:

```
Loading Amazon data from data/raw/laptop_results_2024-11-15.tsv...
Data loaded successfully.
Preprocessing data with standard processing...
Running word frequency analysis and plotting...
Top words by frequency: [('notebook', 59), ('display', 39), ('intel', 38), ...]
Top bigrams by frequency: [(('display', 'fhd'), 10), (('display', 'full_hd'), 10), ...]
Top trigrams by frequency: [(('libre_off', 'pronto', 'alluso'), 5), ('notebook', 'alluminio', 'monitor'), ...]
```

**Visualizations**:

- **Top 10 Most Common Words**:
  ![Top 10 Most Common Words](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Words.png)

- **Top 10 Most Common Bigrams**:
  ![Top 10 Most Common Bigrams](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Bigrams.png)

- **Top 10 Most Common Trigrams**:
  ![Top 10 Most Common Trigrams](./files/1.%20amazon%20prodcut%20search/Top%2010%20Most%20Common%20Trigrams.png)

These images illustrate the most frequent words, bigrams, and trigrams found in the product descriptions.

---

#### 4. Topic Modeling (LDA)

To perform Latent Dirichlet Allocation (LDA) for topic modeling and identify common themes within the product
descriptions, run:

```bash
python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --run_lda --num_topics 5 --passes 15
```

**Console Output**:

```
Loading Amazon data from data/raw/laptop_results_2024-11-15.tsv...
Data loaded successfully.
Preprocessing data with standard processing...
Running LDA topic modeling...
Topic 1: 0.037*"display" + 0.036*"garanzia" + ...
Topic 2: 0.035*"intel" + 0.027*"wifi" + ...
...
```

**Visualization**:

- **LDA Topic Modeling**:
  ![LDA Topic Modeling](./files/1.%20amazon%20prodcut%20search/LDA%20topic%20modeling.png)

This image visualizes the extracted topics and their top words.

---

#### 5. Search Functionality

To search for products related to a specific query and retrieve the top 5 results based on cosine similarity:

```bash
python main_amazon.py --path data/raw/laptop_results_2024-11-15.tsv --run_search --query "Intel Core i7 SSD HD Ram 16Gb" --top_k 5
```

**Console Output**:

```
Loading Amazon data from data/raw/laptop_results_2024-11-15.tsv...
Data loaded successfully.
Preprocessing data with standard processing...
Using non-Spark (original) search engine...
Indexing complete.
Top search results:
Document ID: 5, Score: 0.2735, Description: pc fisso computer desktop intel_core_i7 intel hd masterizz wifi interno ...
Document ID: 173, Score: 0.1756, Description: jumper computer portatile hd display office_365 ...
...
```

The command displays the top 5 search results with their relevance scores and descriptions.

---

#### Notes

- If the `--path` argument is omitted, the program defaults to loading data
  from `data/raw/computer_results_default.tsv`.
- Use the `--top_k` flag to specify the number of search results returned.

---

By following these commands, you can run the complete Amazon Product Search analysis, from data scraping and loading to
word analysis, topic modeling, and search functionality.

---


# Product Search and Flight Delay Prediction

### Table of Contents

1. **Product Search and Flight Delay Prediction**
    - [Project Description](#project-description)
    - [Project Structure](#project-structure)

2. **Amazon Product Search**
    - [Objective](#objective)
    - [Implementation](#implementation)
        - [Web Scraping](#1-web-scraping)
        - [Text Preprocessing](#2-text-preprocessing)
        - [Word Frequency Analysis](#3-word-frequency-analysis)
        - [Search Engine](#4-search-engine)
        - [Topic Modeling](#5-topic-modeling)
    - [Running the Program](#running-the-program)
        - [Setup](#setup)
        - [Commands and Examples](#commands-and-examples)
            - [Scraping Data](#1-scraping-data)
            - [Loading and Preprocessing Existing Data](#2-loading-and-preprocessing-existing-data)
            - [Word Frequency Analysis](#3-word-frequency-analysis)
            - [Topic Modeling (LDA)](#4-topic-modeling-lda)
            - [Search Functionality](#5-search-functionality)
        - [Notes](#notes)

3. **Spark Implementation of Amazon Product Search**
    - [Implementation Steps](#implementation-steps)
        - [Preprocessing with Spark](#1-use-spark-to-preprocess-the-data)
        - [Search Engine with Spark](#2-use-spark-to-build-the-search-engine)
    - [Commands and Examples](#commands-and-examples-1)
        - [Spark Preprocessing and Search](#1-using-spark-to-preprocess-and-search-the-data)
        - [Word, Bigram, and Trigram Frequencies](#2-analyzing-word-bigram-and-trigram-frequencies)
        - [LDA Topic Modeling](#3-running-lda-topic-modeling)

4. **Flight Delay Prediction**
    - [Project Overview](#flight-delay-prediction)
        - [Task Objective](#what-is-the-task)
    - [Commands and Examples](#commands-and-examples-2)
        - [Downloading and Loading Data](#1-downloading-and-loading-data)
        - [Notes on Kaggle API Setup](#notes-on-kaggle-api-setup)
    - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
        - [Delay Distributions](#1-departure-and-arrival-delay-distributions)
        - [Delay and Cancellation Rates by Airline](#2-delay-and-cancellation-rates-by-airline)
        - [Time-Based Patterns](#3-time-based-patterns)
        - [Correlation Heatmap](#4-correlation-heatmap)
        - [Cancellation Reasons](#5-cancellation-reasons)
    - [Handling Missing Values](#3-checking-and-handling-missing-values)
        - [Identifying Missing Values](#identifying-missing-values)
        - [Handling Strategies](#handling-missing-values)
    - [Feature Engineering and Label Preparation](#4-feature-engineering-and-label-preparation)
        - [Feature Selection](#feature-engineering)
        - [Binary Label Preparation](#binary-label-preparation)
    - [Training and Evaluating Models](#5-training-and-evaluating-models)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
    - [Model Details](#model-specific-details)
        - [Logistic Regression](#logistic-regression)
        - [Random Forest](#random-forest)
    - [Model Results and Comparison](#7-results)
        - [Logistic Regression Evaluation](#1-logistic-regression-model-evaluation)
        - [Random Forest Evaluation](#2-random-forest-model-evaluation)
        - [Model Comparison Insights](#3-insights-from-model-comparisons)

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

The following section presents each command clearly with sample console output and
relevant images.

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
  ![Top 10 Most Common Words](files/1.%20amazon%20prodcut%20search%20(without%20Spark)/Top%2010%20Most%20Common%20Words.png)

- **Top 10 Most Common Bigrams**:
  ![Top 10 Most Common Bigrams](files/1.%20amazon%20prodcut%20search%20(without%20Spark)/Top%2010%20Most%20Common%20Bigrams.png)

- **Top 10 Most Common Trigrams**:
  ![Top 10 Most Common Trigrams](files/1.%20amazon%20prodcut%20search%20(without%20Spark)/Top%2010%20Most%20Common%20Trigrams.png)

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
  ![LDA Topic Modeling](files/1.%20amazon%20prodcut%20search%20(without%20Spark)/LDA%20topic%20modeling.png)

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

### Spark Implementation of Amazon Product Search

This project has been implemented on WindowsOS and the spark configuration on Windows has been done based
on [How to Install Apache Spark on Windows](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://phoenixnap.com/kb/install-spark-on-windows-10&ved=2ahUKEwj_hr-un-OJAxUVgv0HHbFaG5sQFnoECBkQAQ&usg=AOvVaw2xn1OY8gD3VZSzm4UY4YVJ)

Assume we have scraped the products using the commands described in [Scraping Data](#1-scraping-data) section.  

So, we have a `.tsv` file in the `data/raw` directory. We call it our "default" scraped data.

To incorporate Spark into the previous task, we can do 2 main things:

1. **Use Spark to Preprocess the Data**  
   Spark's distributed capabilities allow us to preprocess product descriptions efficiently, even for large datasets.
   The preprocessing steps include tokenization, removal of stopwords, lemmatization, and more, as defined in
   the `preprocess_with_pyspark` function.

   **Implementation Steps:**
    - Convert the product dataset into a Spark DataFrame.
    - Apply text preprocessing using Spark's UDFs combined with NLTK.
    - Export the processed descriptions for downstream tasks.

   **Code Reference:** `SparkPreprocessing.py` provides the implementation of these preprocessing functions using
   PySpark and NLTK.

2. **Use Spark to Build the Search Engine**  
   Using Spark for building a search engine involves indexing product descriptions and efficiently handling queries.
   This is done using:
    - TF-IDF for feature extraction.
    - Cosine similarity for query matching.
    - An inverted index for optimizing search performance.

   **Implementation Steps:**
    - Tokenize and preprocess product descriptions.
    - Use Spark MLlib's TF-IDF implementation to transform tokens into numerical vectors.
    - Calculate cosine similarity between query vectors and product vectors to rank search results.

   **Code Reference:** The class `SparkSearchEngine` in `SparkSearchEngine.py` provides an implementation of these
   features.

---

---

### Commands and Examples

Below are some example commands for various tasks, along with sample output and images of results.

#### 1. **Using Spark to Preprocess and Search the Data**

To preprocess and search the data using Spark, use the following command:

```bash
python main_amazon.py --use_pyspark --run_search --keyword "laptop, pc" --query "HP Notebook G9 Intel i3-1215u 6 Core 4,4 Ghz 15,6 Full Hd, Ram 16Gb Ddr4, Ssd Nvme 756Gb M2, Hdmi, Usb 3.0, Wifi, Lan,Bluetooth, Webcam, Windows 11 Professional,Libre Office" --top_k 5
```

**Output:**

```
Loading Amazon data from data/raw/computer_results_default.tsv...
Data loaded successfully.
Preprocessing data with PySpark...
Using Spark for search and indexing...
Top search results:
Document ID: 371, Score: 0.5037001371383667, Description: hp notebook g9 intel i31215u 6 core 44 ghz 156 full hd ram 16gb ddr4 ssd nvme 756gb m2 hdmi usb 30 wifi lan bluetooth webcam window 11 professional libre office
Document ID: 6, Score: 0.5037001371383667, Description: hp notebook g9 intel i31215u 6 core 44 ghz 156 full hd ram 16gb ddr4 ssd nvme 756gb m2 hdmi usb 30 wifi lan bluetooth webcam window 11 professional libre office
Document ID: 655, Score: 0.5037001371383667, Description: hp notebook g9 intel i31215u 6 core 44 ghz 156 full hd ram 16gb ddr4 ssd nvme 756gb m2 hdmi usb 30 wifi lan bluetooth webcam window 11 professional libre office
Document ID: 147, Score: 0.4100857973098755, Description: notebook hp g9 intel i31215u 6 core 44 ghz 156 full hd ram 8gb ddr4 ssd nvme 256gb m2 hdmi usb 30 wifi lan bluetooth webcam window 11 professional libre office
Document ID: 436, Score: 0.4100857973098755, Description: notebook hp g9 intel i31215u 6 core 44 ghz 156 full hd ram 8gb ddr4 ssd nvme 256gb m2 hdmi usb 30 wifi lan bluetooth webcam window 11 professional libre office
```

#### 2. **Analyzing Word, Bigram, and Trigram Frequencies**

To generate the most common words, bigrams, and trigrams, use the following command:

```bash
python main_amazon.py --use_pyspark --keyword "laptop, pc" --plot_frequency --top_words 10 --top_bigrams 10 --top_trigrams 10
```

**Output:**

```
Top 10 Most Common Words**: [('pc', 561), ('ssd', 557), ('ram', 496), ('11', 427), ('pro', 414), ('computer', 397), ('window', 366), ('amd', 342), ('intel', 330), ('core', 319)]
Top 10 Most Common Bigrams**: [(('window', '11'), 276), (('intel', 'core'), 239), (('11', 'pro'), 237), (('pc', 'portatile'), 229), (('amd', 'ryzen'), 222), (('display', '156'), 183), (('core', 'i5'), 167), (('1', 'tb'), 157), (('win', '11'), 151), (('ryzen', '5'), 146)]
Top 10 Most Common Trigrams**: [(('intel', 'core', 'i5'), 163), (('amd', 'ryzen', '5'), 146), (('156', 'full', 'hd'), 137), (('window', '11', 'pro'), 134), (('processore', 'amd', 'ryzen'), 118), (('core', 'i5', '12th'), 103), (('display', '156', 'full'), 103), (('win', '11', 'pro'), 103), (('da', '1', 'tb'), 103), (('window', '11', 'home'), 102)]
```

**Images:**

- ![Top 10 Most Common Words](files/2.%20amazon%20prodcut%20search%20(using%20Spark)/Top%2010%20Most%20Common%20Words.png)
- ![Top 10 Most Common Bigrams](files/2.%20amazon%20prodcut%20search%20(using%20Spark)/Top%2010%20Most%20Common%20Bigrams.png)
- ![Top 10 Most Common Trigrams](files/2.%20amazon%20prodcut%20search%20(using%20Spark)/Top%2010%20Most%20Common%20Trigrams.png)

#### 3. **Running LDA Topic Modeling**

To perform topic modeling on the data using LDA, run the following command:

```bash
python main_amazon.py --use_pyspark --keyword "laptop, pc" --run_lda --num_topics 5 --passes 15
```

**Output:**

```
Loading Amazon data from data/raw/computer_results_default.tsv...
Data loaded successfully.
Preprocessing data with PySpark...
Running LDA topic modeling...
Topic 1: 0.043*"da" + 0.036*"pro" + 0.035*"wifi" + 0.033*"gb" + 0.033*"1" + 0.033*"tb" + 0.031*"intel" + 0.027*"hdmi" + 0.024*"core" + 0.022*"fisso"
Topic 2: 0.038*"1tb" + 0.036*"mini" + 0.030*"mouse" + 0.027*"tastiera" + 0.024*"portatile" + 0.023*"wifi" + 0.023*"pollici" + 0.022*"win" + 0.022*"notebook" + 0.022*"14"
Topic 3: 0.058*"pro" + 0.043*"intel" + 0.037*"hp" + 0.035*"i5" + 0.035*"core" + 0.034*"250" + 0.034*"portatile" + 0.028*"16gb" + 0.026*"office" + 0.026*"g9"
Topic 4: 0.063*"amd" + 0.042*"gb" + 0.037*"processore" + 0.032*"radeon" + 0.032*"ddr5" + 0.032*"ryzen" + 0.031*"8" + 0.027*"home" + 0.027*"display" + 0.021*"mini"
Topic 5: 0.124*"scrivania" + 0.075*"con" + 0.050*"per" + 0.050*"di" + 0.050*"cm" + 0.026*"mouse" + 0.025*"156" + 0.025*"led" + 0.025*"desktop" + 0.025*"gaming"
```

**Image:**

- ![LDA topic modeling](files/2.%20amazon%20prodcut%20search%20(using%20Spark)/LDA%20topic%20modeling.png)

---

## Flight Delay Prediction

This section describes the steps and methodology for predicting flight delays using PySpark, starting from data
preprocessing, exploratory data analysis (EDA), feature engineering, and finally building machine learning models for
binary classification.

What is the task?

Given a dataset of flights, create model predicting whether a flight is delayed by more than 15 minutes.

To be more precise, we will create a model, predicting if a flight is going to **depart** with more than 15 minutes of
delay.

### **Commands and Examples**

---

### 1. **Downloading and Loading Data**

To work with the flight delay dataset, you first need to download and load it into your environment. Follow these steps:

##### Download the Dataset

To download the dataset, execute the following command:

```bash
python main_flight.py download
```

When you pass the `download` action to the `main_flight.py` script, the flight dataset is fetched using the Kaggle API
and stored in the `data/raw` directory. After a successful download, the files are extracted, and you will find the
following in the `data/raw` directory:

- **`flights_sample_3m.csv`**: The actual dataset containing flight delay and cancellation information.
- **`dictionary.html`**: A metadata file with general information about the dataset, including column descriptions and
  data types.

##### Console Output for Successful Download:

After executing the command, you will see this output in the console:

```
Executing action: download
Starting download from Kaggle...
Dataset URL: https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023
Dataset downloaded and extracted to data/
```

##### Loading the Dataset

To load the dataset into your program, execute:

```bash
python main_flight.py load
```

By default, this command loads the file located at `data/raw/flights_sample_3m.csv`. Once downloaded, there is no need
to download the dataset again unless the file is deleted or a new version is required.

---

##### Notes on Kaggle API Setup

The Kaggle API is used to download the dataset. To enable its usage:

1. Create an account on Kaggle and generate an API Token (`kaggle.json`).
2. Place the `kaggle.json` file in the following location:
    - **Linux/Mac**: `~/.kaggle/kaggle.json`
    - **Windows**: `%HOMEPATH%\.kaggle\kaggle.json`

If the `kaggle.json` file is not set up correctly, you will encounter the following error:

```
OSError: Could not find kaggle.json. Make sure it's located in ~/.kaggle. Or use the environment method. See setup instructions at https://github.com/Kaggle/kaggle-api/
```

---

##### Important Notes

- Once downloaded, you do not need to re-download the dataset unless explicitly required.
- The `load` command always loads the `data/raw/flights_sample_3m.csv` file for subsequent analysis or processing.

---

### 2. **Exploratory Data Analysis (EDA)**

The first thing that we do when we deal with a ML problem is analyzing the data.
In this section we perform a thorough investigation of the data and its structure.

#### 1. **Departure and Arrival Delay Distributions**

- **Departure Delays:**
    - ![Distribution of Departure Delays](files/3.%20flight%20prediction/eda/Distribution%20of%20Departure%20Delays.png)
    - ![Distribution of Departure Delays (with mean)](files/3.%20flight%20prediction/eda/Distribution%20of%20Departure%20Delays%20(with%20mean).png)

  Observations:
    - Most flights have minimal or no departure delays, as evident from the sharp peak near zero.
    - A small proportion of flights exhibit significant delays exceeding 500 minutes.
    - The mean departure delay is marked in the plot, providing a central tendency for delays.

- **Arrival Delays:**
    - ![Distribution of Arrival Delays](files/3.%20flight%20prediction/eda/Distribution%20of%20Arrival%20Delays.png)
    - ![Distribution of Arrival Delays (with mean)](files/3.%20flight%20prediction/eda/Distribution%20of%20Arrival%20Delays%20(with%20mean).png)

  Observations:
    - The arrival delay distribution mirrors the departure delay distribution, with most flights clustered around
      minimal or no delay.
    - A small number of flights experience extreme arrival delays.
    - The mean arrival delay is also highlighted for better interpretability.

---

#### 2. **Delay and Cancellation Rates by Airline**

- **Average Delays by Airline:**
    - ![Average Departure and Arrival Delays by Airline](files/3.%20flight%20prediction/eda/Average%20Departure%20and%20Arrival%20Delays%20by%20Airline.png)

  Observations:
    - Airlines such as **Allegiant Air** and **Frontier Airlines** have the highest average delays, while others like *
      *Endeavor Air** exhibit minimal delays.
    - Departure delays tend to be slightly higher than arrival delays for most airlines.

- **Cancellation Rates by Airline:**
    - ![Cancellation Rate by Airline](files/3.%20flight%20prediction/eda/Cancellation%20Rate%20by%20Airline.png)

  Observations:
    - Airlines such as **Frontier Airlines** and **Southwest Airlines** have relatively high cancellation rates compared
      to others.
    - Airlines with low delays may still exhibit notable cancellation rates, highlighting different operational
      challenges.

---

#### 3. **Time-Based Patterns**

- **Delays by Hour of the Day:**
    - ![Average Departure and Arrival Delays by Hour of Day](files/3.%20flight%20prediction/eda/Average%20Departure%20and%20Arrival%20Delays%20by%20Hour%20of%20Day.png)

  Observations:
    - Delays are more frequent during the early morning and late evening hours, likely due to congestion and operational
      inefficiencies.
    - Mid-day flights tend to have shorter delays, reflecting smoother operations during these hours.

- **Delays by Day of the Week:**
    - ![Average Departure and Arrival Delays by Day of the Week](files/3.%20flight%20prediction/eda/Average%20Departure%20and%20Arrival%20Delays%20by%20Day%20of%20the%20Week.png)

  Observations:
    - Delays are higher towards the end of the week, peaking on **Thursdays** and **Fridays**.
    - This could be attributed to higher travel volumes and operational strain during these days.

---

#### 4. **Correlation Heatmap**

- ![Correlation Heatmap for Numerical Features](files/3.%20flight%20prediction/eda/Correlation%20Heatmap%20for%20Numerical%20Features.png)

  Observations:
    - Departure and arrival delays are highly correlated, indicating that departure delays directly contribute to
      arrival delays.
    - Features like **distance** and **weather-related delays** show moderate correlations with the delay metrics,
      providing insights into contributing factors.

---

#### 5. **Cancellation Reasons**

- ![Distribution of Cancellation Reasons](files/3.%20flight%20prediction/eda/Distribution%20of%20Cancellation%20Reasons.png)

  Observations:
    - The most common cancellation reasons include **carrier-related issues** and **weather conditions**.
    - Security-related cancellations are comparatively rare, reflecting their infrequent occurrence.

---

---

### 6. **Train and Test Set Distribution**

After splitting the dataset into train (80%) and test (20%) sets, it's important to verify the distribution of labels in
both subsets to ensure consistency and fairness in model evaluation. Note that due to limited computational resources,
only 50% of the dataset was used for this analysis.

#### **Train Set Label Distribution**

```
+-----+------+
|label| count|
+-----+------+
|    1|206406|
|    0|996014|
+-----+------+
```

- The training set contains **206,406 flights labeled as delayed (label = 1)** and **996,014 flights labeled as
  non-delayed (label = 0)**.
- This results in a **17.2% delay rate**, with a roughly consistent proportion of delayed to non-delayed flights.

#### **Test Set Label Distribution**

```
+-----+------+
|label| count|
+-----+------+
|    1| 51390|
|    0|249059|
+-----+------+
```

- The test set contains **51,390 delayed flights (label = 1)** and **249,059 non-delayed flights (label = 0)**.
- The delay rate is similar to that of the training set, ensuring consistency in the distribution.

---

### Key Insights:

1. **Consistent Proportions:** Both train and test sets have nearly identical proportions of delayed to non-delayed
   flights, ensuring that the model is trained and evaluated under comparable conditions.
2. **Impact of Imbalance:** Although the dataset is imbalanced, the consistent proportions across subsets reduce the
   potential impact of this imbalance on model training and evaluation. Many machine learning algorithms, particularly
   those using cross-validation and metrics like AUC, are resilient to such imbalances. Moreover, this is actually what
   we see (or we hope) to happen in the real-world. most of the flights are departed on time and only a small portion of
   them are delayed.

---

### 3. Checking and Handling Missing Values

To ensure data integrity, the script performs the following actions for identifying and managing missing values:

#### **Identifying Missing Values**

You can identify missing data by using the `check_missing` command:

```bash
python main_flight.py load check_missing
```

This command computes and displays the number of `NaN` or `null` values for each column in the dataset. For instance,
the output below indicates the presence of missing values in several columns such
as `DEP_TIME`, `DEP_DELAY`, `ARR_TIME`, and specific delay-related columns like `DELAY_DUE_CARRIER`:

```
Executing action: check_missing
+-------+-------+-----------+------------+--------+---------+------+-----------+----+---------+------------+--------+---------+--------+----------+---------+-------+------------+--------+---------+---------+-----------------+--------+----------------+------------+--------+--------+-----------------+-----------------+-------------+------------------+-----------------------+
|FL_DATE|AIRLINE|AIRLINE_DOT|AIRLINE_CODE|DOT_CODE|FL_NUMBER|ORIGIN|ORIGIN_CITY|DEST|DEST_CITY|CRS_DEP_TIME|DEP_TIME|DEP_DELAY|TAXI_OUT|WHEELS_OFF|WHEELS_ON|TAXI_IN|CRS_ARR_TIME|ARR_TIME|ARR_DELAY|CANCELLED|CANCELLATION_CODE|DIVERTED|CRS_ELAPSED_TIME|ELAPSED_TIME|AIR_TIME|DISTANCE|DELAY_DUE_CARRIER|DELAY_DUE_WEATHER|DELAY_DUE_NAS|DELAY_DUE_SECURITY|DELAY_DUE_LATE_AIRCRAFT|
+-------+-------+-----------+------------+--------+---------+------+-----------+----+---------+------------+--------+---------+--------+----------+---------+-------+------------+--------+---------+---------+-----------------+--------+----------------+------------+--------+--------+-----------------+-----------------+-------------+------------------+-----------------------+
|      0|      0|          0|           0|       0|        0|     0|          0|   0|        0|           0|   77615|    77644|   78806|     78806|    79944|  79944|           0|   79942|    86198|        0|          2920860|       0|              14|       86198|   86198|       0|          2466137|          2466137|      2466137|           2466137|                2466137|
+-------+-------+-----------+------------+--------+---------+------+-----------+----+---------+------------+--------+---------+--------+----------+---------+-------+------------+--------+---------+---------+-----------------+--------+----------------+------------+--------+--------+-----------------+-----------------+-------------+------------------+-----------------------+
```

---

#### **Handling Missing Values**

After identifying missing data, the `handle_missing_values` function is executed to clean and preprocess the dataset.
This function applies the following strategies:

1. **Imputation for Delays:**
    - Missing values in `DEP_DELAY` and `ARR_DELAY` are imputed with `0.0`, assuming that missing delay information
      indicates no delay.
    - **Example:** If a row lacks departure delay data, it will be filled with `0.0`.

2. **Dropping Rows with Critical Missing Information:**
    - Rows with missing values in essential columns such as `AIRLINE`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DISTANCE`
      ,etc. are dropped.
    - These columns are fundamental for modeling delays; missing values here could compromise the quality of
      predictions.

3. **Imputation for Delay-Related Columns:**
    - Columns representing specific types of
      delays (`DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`)
      are imputed with `0.0`, assuming no delay occurred if the value is missing.
    - **Example:** If `DELAY_DUE_CARRIER` is missing, it will be set to `0.0`.

---

### **4. Feature Engineering and Label Preparation**

After addressing missing values, the next steps involve preparing the dataset for machine learning by performing *
*feature engineering** and **binary label creation**:

#### **Feature Engineering**

1. **Feature Selection:**
    - Relevant features for delay prediction are selected, including:
        - **Time-related features:** `CRS_DEP_TIME` (scheduled departure time) and `DISTANCE` (flight distance).
        - **Delay-related causes:** `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`,
          and `DELAY_DUE_LATE_AIRCRAFT`.
        - **Categorical features:** `AIRLINE`, `ORIGIN`, and `DEST`.

2. **Categorical Feature Encoding:**
    - Categorical columns (`AIRLINE`, `ORIGIN`, `DEST`) are processed using:
        - **StringIndexer:** Converts categorical string values to numerical indices.
        - **OneHotEncoder:** Converts the indexed values into binary vectors for use in machine learning.

3. **Feature Assembly:**
    - The selected numerical features and the encoded categorical vectors are combined into a single feature vector
      using `VectorAssembler`.
    - This unified representation (`features`) is essential for model training.

**But why these features (columns) are selected and others are rejected?**

The reason is that these features provide a balance of **temporal, spatial, and causal information**, which are crucial
for predicting delays.

- **Time and distance** account for operational and logistical factors affecting delays.
- **Categorical features** capture unique characteristics tied to airlines and airports.
- The **delay-related columns** directly explain known factors contributing to delays, making them predictive.

To know why each of these features are selected from my POV:

- **Time-related Features:**
    - `CRS_DEP_TIME` (scheduled departure time): Helps capture patterns in delays based on time of day.
    - `DISTANCE` (flight distance): Longer flights may face different delay dynamics compared to shorter flights.

- **Delay-related Causes:**
    - `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, and `DELAY_DUE_LATE_AIRCRAFT`:
      These columns provide detailed reasons for delays, directly informing the prediction model.

- **Categorical Features:**
    - `AIRLINE`, `ORIGIN`, and `DEST`: These features capture airline-specific, origin-specific, and
      destination-specific patterns in delays.

### **Binary Label Preparation**

Because there is no column in the dataset indicating the "greater or equal to 15 minutes of departure delay", we need to
create this column. How?

We create another column for each row named `label` which is True (1) if the departure was delayed by 15 minutes and
False (0) else.
This binary label enables the model to focus on identifying significant delays, simplifying the classification task into
a binary decision-making problem.

---

### **5. Training and Evaluating Models**

To train and evaluate the Logistic Regression and Random Forest models:

**Logistic Regression:**

```bash
python main_flight.py load train_evaluate_logistic_regression
```

**Random Forest:**

```bash
python main_flight.py load train_evaluate_random_forest
```

---

#### Training and Evaluating Models

But how the Logistic Regression and Random Forest models are trained, tuned, and evaluated to predict flight delays?

#### **Training Process**

When any of the `train_evaluate` commands is executed, this is what is happening:

1. **Data Preparation:**
    - The dataset is preprocessed by handling missing values, performing feature engineering, and preparing a binary
      label column indicating whether a flight is delayed by more than 15 minutes.

2. **Train-Test Split:**
    - The data is split into training and testing sets (80-20) to train the models and evaluate their performance on
      unseen data.

3. **Model Tuning:**
    - Both models are tuned using a hyperparameter grid search:
        - Logistic Regression:
            - Regularization parameter (`regParam`).
            - Elastic Net mixing ratio (`elasticNetParam`).
        - Random Forest:
            - Number of trees (`numTrees`).
            - Maximum depth of trees (`maxDepth`).

4. **Cross-Validation:**
    - Five-fold cross-validation is applied during model training to find the best hyperparameter settings based on the
      Area Under the ROC Curve (AUC).

5. **Final Training:**
    - The best-performing model from cross-validation is used to make predictions on the test dataset.

Then with the best model, we do the evaluation as following

---

### **6. Evaluation Process**

1. **Metrics Computed:**
    - Key evaluation metrics include:
        - **AUC (Area Under the Curve):** Measures the ability to distinguish between delayed and non-delayed flights.
        - **Accuracy:** Percentage of correct predictions.
        - **Precision:** Proportion of predicted delays that are actual delays.
        - **Recall:** Proportion of actual delays that are correctly predicted.
        - **F1-Score:** Harmonic mean of precision and recall.
    - A confusion matrix is generated to summarize the performance.

2. **ROC Curve:**
    - The ROC curve is plotted to visualize the model’s true positive rate (TPR) versus the false positive rate (FPR)
      across different thresholds.

3. **Random Forest Feature Importance:**
    - For the Random Forest model, feature importance is computed to identify which factors contribute most to
      predicting flight delays.

---

### 7. Neural Network Model

A neural network model was introduced for flight delay prediction, providing a more flexible and powerful approach to
capture complex, non-linear relationships between features and labels.

This model training and evaluation can be executed with the following command:

```bash
python main_flight.py load train_evaluate_neural_network
```

#### **Model Architecture**

- **Input Layer:** Takes in the feature vector (`input_dim`).
- **Hidden Layers:**
    - First layer: 128 neurons with ReLU activation.
    - Second layer: 64 neurons with ReLU activation.
- **Output Layer:** A single neuron with a Sigmoid activation function to produce probabilities for binary
  classification.
- **Loss Function:** Binary Cross-Entropy Loss (BCELoss), which is suitable for binary classification tasks.
- **Optimizer:** Adam optimizer with a learning rate of `0.001`.

#### **Training Details**

- The model was trained for **10 epochs** with a batch size of 32.
- **Loss values per epoch:**
  ```
  Epoch 1/10, Loss: 0.21436312794685364
  Epoch 2/10, Loss: 0.12203751504421234
  Epoch 3/10, Loss: 0.1563258320093155
  Epoch 4/10, Loss: 0.203612819314003
  Epoch 5/10, Loss: 0.14350281655788422
  Epoch 6/10, Loss: 0.14558719098567963
  Epoch 7/10, Loss: 0.13873544335365295
  Epoch 8/10, Loss: 0.1565149873495102
  Epoch 9/10, Loss: 0.29420629143714905
  Epoch 10/10, Loss: 0.16712436079978943
  ```

---

A little literature about the models we used in this project.
What are they used and how they should be evaluated?

#### **Model-Specific Details**

- **Logistic Regression:**
    - Suitable for linear relationships between features and the label.
    - AUC and other metrics are used to evaluate its ability to classify delayed and non-delayed flights.

- **Random Forest:**
    - Effective for capturing non-linear relationships.
    - Provides insights into feature importance, highlighting key factors influencing flight delays.

- **Neural Network:**
    - The neural network handles non-linear relationships between features more effectively than Logistic Regression or
      Random Forest.
    - Its flexibility in architecture allows for future extensions, such as incorporating additional features or
      fine-tuning hyperparameters.

---

### **7. Results**

Finally, it is time to evaluate and compare our Logistic Regression and Random Forest models

#### 1. **Logistic Regression Model Evaluation**

- **Performance Metrics:**
  ```
  AUC: 0.9235
  Accuracy: 91.60%
  Precision: 92.17%
  Recall: 91.60%
  F1-Score: 90.50%
  ```
    - The high **AUC (0.9235)** indicates that the Logistic Regression model is effective at distinguishing between
      delayed and non-delayed flights.
    - The **Accuracy (91.60%)** suggests the model is highly accurate at predicting delays.
    - **Precision (92.17%)** shows that the model is good at minimizing false positives, meaning most flights predicted
      as delayed are actually delayed.
    - The **Recall (91.60%)** reflects that the model correctly identifies most delayed flights, although a small
      fraction is missed.
    - The **F1-Score (90.50%)** balances precision and recall, confirming the model's strong overall performance.

- **Confusion Matrix:**
  ```
  +-----+----------+------+
  |label|prediction| count|
  +-----+----------+------+
  |    0|       0.0|248525|
  |    0|       1.0|   534|
  |    1|       0.0| 24714|
  |    1|       1.0| 26676|
  +-----+----------+------+
  ```
    - True Negatives (248,525): Non-delayed flights correctly identified.
    - False Positives (534): Flights incorrectly predicted as delayed.
    - True Positives (26,676): Delayed flights correctly identified.
    - False Negatives (24,714): Delayed flights incorrectly predicted as non-delayed.
    - Interpretation:
        - The model performs well with very few false positives but struggles slightly with false negatives, which could
          impact its ability to catch all delays.

- **ROC Curve Visualization:**
    - ![Logistic Regression ROC Curve](files/3.%20flight%20prediction/roc/lr%20roc%20curve.png)
    - The ROC curve shows strong separation between true positives and false positives, reflecting excellent
      performance.

---

#### 2. **Random Forest Model Evaluation**

- **Performance Metrics:**
  ```
  AUC: 0.9229
  Accuracy: 90.45%
  Precision: 91.04%
  Recall: 90.45%
  F1-Score: 88.99%
  ```
    - The **AUC (0.9229)** is nearly identical to Logistic Regression, indicating similar capability in distinguishing
      delays.
    - The **Accuracy (90.45%)** is slightly lower than Logistic Regression.
    - **Precision (91.04%)** remains high, but lower than Logistic Regression, showing more false positives.
    - **Recall (90.45%)** is also slightly lower, meaning some delayed flights are missed.
    - The **F1-Score (88.99%)** suggests a slight drop in overall balance compared to Logistic Regression.

- **Confusion Matrix:**
  ```
  +-----+----------+------+
  |label|prediction| count|
  +-----+----------+------+
  |    0|       0.0|248201|
  |    0|       1.0|   858|
  |    1|       0.0| 27829|
  |    1|       1.0| 23561|
  +-----+----------+------+
  ```
    - True Negatives (248,201): Non-delayed flights correctly identified.
    - False Positives (858): More flights incorrectly predicted as delayed compared to Logistic Regression.
    - True Positives (23,561): Fewer delayed flights correctly identified compared to Logistic Regression.
    - False Negatives (27,829): Higher number of delayed flights missed compared to Logistic Regression.
    - Interpretation:
        - Random Forest is slightly less precise and has a higher false negative rate than Logistic Regression, leading
          to fewer delayed flights being captured.

- **Feature Importance Visualization:**
    - The Random Forest model allows interpretation of feature importance, which can guide understanding of the factors
      most predictive of delays.

---

#### 3. **NN Model Evaluation Metrics**

The neural network was evaluated on the test dataset, and the following metrics were observed:

- **AUC:** 0.926
- **Accuracy:** 95.51%
- **Precision:** 95.24%
- **Recall:** 77.46%
- **F1-Score:** 85.43%

---

#### **Interpretation of Results**

- **AUC (0.926):** Indicates excellent ability to distinguish between delayed and non-delayed flights. The model
  effectively ranks predictions across thresholds.
- **Accuracy (95.51%):** The model accurately predicts delay status in most cases, reflecting its robustness.
- **Precision (95.24%):** High precision shows that the model minimizes false positives, meaning it seldom misclassifies
  non-delayed flights as delayed.
- **Recall (77.46%):** While slightly lower than precision, recall indicates the model captures most delayed flights but
  misses some.
- **F1-Score (85.43%):** Balances precision and recall, showing the model is reliable overall.

---

#### 4. **Insights from Model Comparisons**

- Logistic Regression outperforms Random Forest slightly in terms of accuracy, precision, recall, and F1-score.
- Both models exhibit high AUC values, confirming their effectiveness in separating delayed from non-delayed flights.
- Logistic Regression has a lower false negative rate, making it preferable when identifying all delayed flights is
  critical.
- Random Forest provides feature importance insights, which can be invaluable for further analysis and domain-specific
  interventions.

---
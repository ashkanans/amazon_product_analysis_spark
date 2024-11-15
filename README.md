
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
import re

from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, struct, collect_list, array

# Initialize Spark session
spark = SparkSession.builder.appName("SparkSearchEngine").getOrCreate()


class SparkSearchEngine:
    def __init__(self, min_score_threshold=0.05, num_features=10000):
        self.min_score_threshold = min_score_threshold
        self.num_features = num_features
        self.inverted_index = None
        self.document_tfidf = None

    def tokenize(self, text):
        # Simple tokenizer that lowercases and removes punctuation
        text = re.sub(r'\W+', ' ', text.lower())
        tokens = text.split()
        return tokens

    def build_inverted_index(self, documents):
        """
        Build an inverted index with term positions for proximity search.
        """
        # Convert to Spark DataFrame
        rdd = spark.sparkContext.parallelize([(doc_id, self.tokenize(text)) for doc_id, text in documents.items()])
        df = rdd.toDF(["doc_id", "tokens"])

        # Explode tokens with positions for proximity search
        tokens_with_pos = df.withColumn("tokens_with_pos", explode(array(
            *[struct(lit(i).alias("pos"), col("tokens")[i].alias("token")) for i in
              range(0, df.selectExpr("size(tokens)").head()[0])])))

        # Build inverted index by grouping by token and collecting positions
        inverted_index = tokens_with_pos.groupBy("token").agg(
            collect_list(struct("doc_id", "tokens_with_pos.pos")).alias("posting_list"))

        # Save to file (optional)
        inverted_index.write.json("inverted_index.json")

        self.inverted_index = inverted_index

    def calculate_tfidf(self, documents):
        """
        Calculate TF-IDF vectors for each document.
        """
        # Convert documents to Spark DataFrame
        rdd = spark.sparkContext.parallelize(
            [(doc_id, ' '.join(self.tokenize(text))) for doc_id, text in documents.items()])
        df = rdd.toDF(["doc_id", "text"])

        # Apply hashing to represent terms numerically
        hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=self.num_features)
        featurized_data = hashingTF.transform(df)

        # Calculate IDF
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(featurized_data)
        rescaled_data = idf_model.transform(featurized_data)

        self.document_tfidf = rescaled_data.select("doc_id", "features")
        self.document_tfidf.write.parquet("document_tfidf.parquet")

    def calculate_cosine_similarity(self, query, top_k=5):
        """
        Calculate cosine similarity between the query and each document.
        """
        # Tokenize and process the query
        query_tokens = self.tokenize(query)
        query_df = spark.createDataFrame([(0, query_tokens)], ["doc_id", "tokens"])

        # Transform query tokens into TF-IDF
        hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=self.num_features)
        query_featurized = hashingTF.transform(query_df)
        idf_model = IDF(inputCol="rawFeatures", outputCol="features")
        query_features = idf_model.transform(query_featurized).select("features").head()[0]

        # Calculate cosine similarity with each document
        cosine_similarity_udf = udf(
            lambda v1, v2: float(v1.dot(v2)) / (float(v1.norm(2)) * float(v2.norm(2))) if v1.norm(2) != 0 and v2.norm(
                2) != 0 else 0.0, FloatType())
        similarities = self.document_tfidf.withColumn("similarity",
                                                      cosine_similarity_udf(col("features"), lit(query_features)))

        # Filter and sort results based on similarity score
        results = similarities.filter(col("similarity") >= self.min_score_threshold).orderBy(
            col("similarity").desc()).limit(top_k)

        # Collect and return the top_k results
        return results.select("doc_id", "similarity").collect()

    def search(self, query, top_k=5):
        return self.calculate_cosine_similarity(query, top_k=top_k)

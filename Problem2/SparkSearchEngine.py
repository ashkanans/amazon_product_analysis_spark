import re
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, struct, collect_list, array, udf
from pyspark.sql.types import FloatType

# Initialize Spark session
spark = (SparkSession.builder.appName("SparkSearchEngine")
         .getOrCreate())
spark.sparkContext.setCheckpointDir("checkpoint_dir")


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

        # Flatten the tokens_with_pos struct into separate columns
        tokens_with_pos = tokens_with_pos.select("doc_id", col("tokens_with_pos.token"), col("tokens_with_pos.pos"))

        # Build inverted index by grouping by token and collecting positions
        inverted_index = tokens_with_pos.groupBy("token").agg(
            collect_list(struct("doc_id", "pos")).alias("posting_list"))

        # Save to file (optional)
        # inverted_index.write.csv("inverted_index.csv")

        self.inverted_index = inverted_index

    def calculate_tfidf(self, documents):
        """
        Calculate TF-IDF vectors for each document.
        """
        # Convert documents to Spark DataFrame
        rdd = spark.sparkContext.parallelize(
            [(doc_id, ' '.join(self.tokenize(text))) for doc_id, text in documents.items()])
        df = rdd.toDF(["doc_id", "text"])

        # Tokenize text to create 'tokens' column
        tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
        tokenized_df = tokenizer.transform(df)

        # Apply hashing to represent terms numerically
        hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=self.num_features)
        featurized_data = hashingTF.transform(tokenized_df)

        # Calculate IDF and store the model
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        self.idf_model = idf.fit(featurized_data)  # Fit and store the model
        rescaled_data = self.idf_model.transform(featurized_data)

        self.document_tfidf = rescaled_data.select("doc_id", "features")

    def calculate_cosine_similarity(self, query, top_k=5):
        """
        Calculate cosine similarity between the query and each document.
        """
        # Tokenize and process the query
        query_tokens = self.tokenize(query)
        query_df = spark.createDataFrame([(0, query_tokens)], ["doc_id", "tokens"])

        # Transform query tokens into TF-IDF using the stored IDF model
        hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=self.num_features)
        query_featurized = hashingTF.transform(query_df)
        query_features = self.idf_model.transform(query_featurized).select("features").head()[0]

        # Broadcast the query features
        query_features_broadcast = spark.sparkContext.broadcast(query_features)

        # Define UDF to calculate cosine similarity
        cosine_similarity_udf = udf(
            lambda v: float(v.dot(query_features_broadcast.value)) /
                      (float(v.norm(2)) * float(query_features_broadcast.value.norm(2)))
            if v.norm(2) != 0 and query_features_broadcast.value.norm(2) != 0 else 0.0,
            FloatType()
        )

        # Apply the UDF to calculate similarity for each document
        similarities = self.document_tfidf.withColumn("similarity", cosine_similarity_udf(col("features")))

        # Filter and sort results based on similarity score
        results = similarities.filter(col("similarity") >= self.min_score_threshold).orderBy(
            col("similarity").desc()).limit(top_k)

        # Collect and return the top_k results
        return results.select("doc_id", "similarity").collect()

    def search(self, query, top_k=5):
        return self.calculate_cosine_similarity(query, top_k=top_k)

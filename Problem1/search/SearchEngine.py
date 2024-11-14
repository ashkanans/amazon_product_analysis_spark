from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from Problem1.text_processing.TextPreprocessor import TextPreprocessor


class SearchEngine:
    def __init__(self, min_score_threshold=0.05):
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.documents = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            max_df=0.9,  # Ignore terms that appear in more than 80% of documents
            min_df=2  # Ignore terms that appear in fewer than 2 documents
        )
        self.tfidf_matrix = None
        self.min_score_threshold = min_score_threshold
        self.text_preprocessor = TextPreprocessor()

    def preprocess_documents(self, documents):
        """
        Preprocesses the documents using TextPreprocessor.
        :param documents: dict of doc_id -> text
        :return: dict of doc_id -> preprocessed text
        """
        preprocessed_documents = {}
        for doc_id, text in documents.items():
            # Tokenize and process each document
            tokens = self.text_preprocessor.preprocess_text(text)
            # Join tokens back into a single string for TF-IDF vectorizer
            preprocessed_documents[doc_id] = ' '.join(tokens)
        return preprocessed_documents

    def index_documents(self, documents):
        """
        Build the TF-IDF matrix from the preprocessed documents.
        :param documents: dict of doc_id -> text
        """
        # Preprocess documents and prepare them for indexing
        preprocessed_documents = self.preprocess_documents(documents)

        # Remove duplicates by converting to a set of (id, text) pairs, then back to a dict
        unique_texts = {}
        for doc_id, text in preprocessed_documents.items():
            if text not in unique_texts.values():
                unique_texts[doc_id] = text

        self.documents = unique_texts  # Store only unique preprocessed documents

        # Build TF-IDF matrix using the unique preprocessed texts
        texts = list(unique_texts.values())
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print("Indexing complete.")

    def calculate_cosine_similarity(self, query):
        """
        Calculate cosine similarity between the query and each document.
        :param query: str, the query text
        :return: list of (doc_id, score), sorted by score descending
        """
        # Preprocess the query before vectorizing
        query_tokens = self.text_preprocessor.preprocess_text(query)
        query_text = ' '.join(query_tokens)

        query_vector = self.tfidf_vectorizer.transform([query_text])

        # Calculate cosine similarity scores
        cosine_similarities = query_vector * self.tfidf_matrix.T
        scores = cosine_similarities.toarray()[0]

        # Apply minimum score threshold
        ranked_docs = sorted(
            ((doc_id, score) for doc_id, score in zip(self.documents.keys(), scores) if
             score >= self.min_score_threshold),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked_docs

    def search(self, query, top_k=5):
        """
        Process the query and return the top K documents.
        :param query: str, the input query
        :param top_k: int, number of top documents to return
        :return: list of (doc_id, score)
        """
        ranked_docs = self.calculate_cosine_similarity(query)
        return ranked_docs[:top_k]

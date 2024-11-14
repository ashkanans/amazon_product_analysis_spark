from collections import Counter

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from matplotlib import pyplot as plt


class LDAAnalyzer:
    def __init__(self, preprocessed_data, num_topics=5, passes=15, random_state=42):
        """
        Initialize with preprocessed data and LDA parameters.
        :param preprocessed_data: List of lists, where each inner list contains tokens of a product description.
        :param num_topics: int, the number of topics to extract.
        :param passes: int, the number of passes through the corpus during training.
        :param random_state: int, random seed for reproducibility.
        """
        self.preprocessed_data = preprocessed_data
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def prepare_corpus(self):
        """
        Prepares the corpus and dictionary needed for LDA.
        """
        # Create a dictionary representation of the documents.
        self.dictionary = corpora.Dictionary(self.preprocessed_data)

        # Filter out rare and common words.
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)

        # Create a corpus: list of bag-of-words representation of each document.
        self.corpus = [self.dictionary.doc2bow(description) for description in self.preprocessed_data]

    def run_lda(self):
        """
        Run the LDA model on the corpus.
        :return: LDA model
        """
        # Prepare the corpus and dictionary if not already done
        if self.corpus is None or self.dictionary is None:
            self.prepare_corpus()

        # Train the LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=self.passes,
            alpha='auto'
        )
        return self.lda_model

    def display_topics(self, num_words=10):
        """
        Display the topics and their top words.
        :param num_words: int, the number of words to show for each topic.
        """
        if self.lda_model is None:
            print("LDA model has not been trained yet.")
            return

        for i, topic in self.lda_model.print_topics(num_words=num_words):
            print(f"Topic {i + 1}: {topic}")

    def visualize_topics(self):
        """
        Visualize the topics using matplotlib, if you have pyLDAvis you can also use it for interactive visualization.
        """
        if self.lda_model is None:
            print("LDA model has not been trained yet.")
            return

        # Prepare data for visualization
        topics = self.lda_model.show_topics(formatted=False)
        data_flat = [word for description in self.preprocessed_data for word in description]
        counter = Counter(data_flat)

        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i, weight, counter[word]])

        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

        # Plot the topics
        fig, axes = plt.subplots(1, self.num_topics, figsize=(15, 5), sharey=True, dpi=120)
        for i, ax in enumerate(axes.flatten()):
            ax.barh(df.loc[df.topic_id == i, 'word'], df.loc[df.topic_id == i, 'importance'], color='blue')
            ax.set_title(f'Topic {i + 1}')
            ax.invert_yaxis()
            ax.tick_params(axis='y', which='major', labelsize=9)
        plt.tight_layout()
        plt.show()

from collections import Counter

import matplotlib.pyplot as plt
from nltk import bigrams, trigrams


class WordFrequencyAnalyzer:
    def __init__(self, preprocessed_data):
        """
        Initialize with a list of preprocessed product descriptions.
        :param preprocessed_data: List of lists, where each inner list contains tokens of a product description.
        """
        self.preprocessed_data = preprocessed_data
        self.word_freq = None
        self.bigram_freq = None
        self.trigram_freq = None

    def calculate_word_frequency(self):
        """
        Calculate the frequency of each word across all descriptions.
        :return: Counter object with word frequencies.
        """
        # Flatten the list of tokens
        all_words = [word for description in self.preprocessed_data for word in description]
        self.word_freq = Counter(all_words)
        return self.word_freq

    def calculate_bigram_frequency(self):
        """
        Calculate the frequency of each bigram (pair of words) across all descriptions.
        :return: Counter object with bigram frequencies.
        """
        # Generate bigrams from all descriptions
        all_bigrams = [bigram for description in self.preprocessed_data for bigram in bigrams(description)]
        self.bigram_freq = Counter(all_bigrams)
        return self.bigram_freq

    def calculate_trigram_frequency(self):
        """
        Calculate the frequency of each trigram (triple of words) across all descriptions.
        :return: Counter object with trigram frequencies.
        """
        # Generate trigrams from all descriptions
        all_trigrams = [trigram for description in self.preprocessed_data for trigram in trigrams(description)]
        self.trigram_freq = Counter(all_trigrams)
        return self.trigram_freq

    def plot_top_words(self, top_n=10):
        """
        Plot the top N words by frequency with labels displayed above each bar.
        :param top_n: int, the number of top words to display.
        """
        if not self.word_freq:
            self.calculate_word_frequency()

        most_common_words = self.word_freq.most_common(top_n)
        words, frequencies = zip(*most_common_words)

        plt.figure(figsize=(8, 4))
        bars = plt.bar(words, frequencies)
        plt.title(f'Top {top_n} Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')

        # Rotate labels, align them, and set a smaller font size
        plt.xticks(rotation=45, ha="right", fontsize=9)

        # Add labels above each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom', fontsize=8)

        # Add tight layout for better spacing
        plt.tight_layout()

        plt.show(block=True)

    def plot_top_bigrams(self, top_n=10):
        """
        Plot the top N bigrams by frequency with labels displayed above each bar.
        :param top_n: int, the number of top bigrams to display.
        """
        if not self.bigram_freq:
            self.calculate_bigram_frequency()

        most_common_bigrams = self.bigram_freq.most_common(top_n)
        bigrams, frequencies = zip(*most_common_bigrams)
        bigram_labels = [' '.join(bigram) for bigram in bigrams]

        # Adjust figure size to make the chart smaller
        plt.figure(figsize=(8, 4))
        bars = plt.bar(bigram_labels, frequencies)
        plt.title(f'Top {top_n} Most Common Bigrams')
        plt.xlabel('Bigrams')
        plt.ylabel('Frequency')

        # Rotate labels, align them, and set a smaller font size
        plt.xticks(rotation=45, ha="right", fontsize=9)

        # Add labels above each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom', fontsize=8)

        # Add tight layout for better spacing
        plt.tight_layout()

        plt.show(block=True)

    def plot_top_trigrams(self, top_n=10):
        """
        Plot the top N trigrams by frequency with labels displayed above each bar.
        :param top_n: int, the number of top trigrams to display.
        """
        if not self.trigram_freq:
            self.calculate_trigram_frequency()

        most_common_trigrams = self.trigram_freq.most_common(top_n)
        trigrams, frequencies = zip(*most_common_trigrams)
        trigram_labels = [' '.join(trigram) for trigram in trigrams]

        plt.figure(figsize=(8, 4))
        bars = plt.bar(trigram_labels, frequencies)
        plt.title(f'Top {top_n} Most Common Trigrams')
        plt.xlabel('Trigrams')
        plt.ylabel('Frequency')

        # Rotate labels, align them, and set a smaller font size
        plt.xticks(rotation=45, ha="right", fontsize=9)

        # Add labels above each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom', fontsize=8)

        # Add tight layout for better spacing
        plt.tight_layout()

        plt.show(block=True)

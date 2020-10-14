import csv
from nltk.stem import WordNetLemmatizer


class Utilities(object):
    def __init__(self):
        self.sentiment_classes = ['negative', 'neutral', 'positive']
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def read_from_csv(self, file_path):
        data = []

        with open(file_path, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data.append(row)

        return data

    def save_list_as_csv(self, data_list, file_path):
        with open(file_path, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(data_list)

    def tokenize(self, sentence):
        from nltk.tokenize import TweetTokenizer
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(sentence)

        return tokens

    def get_lemma(self, sentences):
        lemmatized = {}
        tokens = self.tokenize(sentences)
        for token in tokens:
            results = self.wordnet_lemmatizer.lemmatize(token)
            lemmatized[token] = results

        return lemmatized
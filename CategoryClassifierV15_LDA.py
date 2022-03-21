import ssl
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CategoryClassifier:
    def __init__(self, trainingData=None):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # nltk.download("wordnet")
        # nltk.download("stopwords")
        # nltk.download("punkt")
        # nltk.download("omw-1.4")
        self.__trainingData = trainingData

        if not self.__trainingData == None:
            self.__steamedTrainingData = []
            for data in self.__trainingData:
                self.__steamedTrainingData.append([self.cleanUp(data[0]), data[1]])

        self.__vectorizer = TfidfVectorizer()
        self.__tfidf_train = self.__vectorizer.fit_transform(
            [sen[0] for sen in self.__steamedTrainingData]
        )

        self.__model = LinearDiscriminantAnalysis()
        self.__model.fit(
            self.__tfidf_train.toarray(), [sen[1] for sen in self.__steamedTrainingData]
        )

    def cleanUp(self, sen):
        sen = sen.lower().replace(".", " ").replace(",", " ")

        a = [word for word in sen.split() if word not in stopwords.words("english")]
        no_stopwords_txt = " ".join(a)

        snow = SnowballStemmer("english")
        b = [snow.stem(word) for word in word_tokenize(no_stopwords_txt)]
        stemmed_txt = " ".join(b)

        wordnet_lemmatizer = WordNetLemmatizer()
        c = [wordnet_lemmatizer.lemmatize(word) for word in stemmed_txt.split(" ")]
        lemmatized_txt = " ".join(c)

        return lemmatized_txt

    def classify(self, sentence):
        clean = self.cleanUp(sentence)
        vector = self.featureExtractor(clean)
        return self.__model.predict(vector)[0]

    def featureExtractor(self, sen):
        tf_feature_test = self.__vectorizer.transform([sen])
        return tf_feature_test


if __name__ == "__main__":
    pass

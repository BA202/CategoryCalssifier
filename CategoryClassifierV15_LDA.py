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
    # from DataHandler.DataHandler import DataHandler
    # from ModelReport.ModelReport import ModelReport
    # from sklearn.model_selection import train_test_split

    # my_data_handler = DataHandler()
    # category_list = my_data_handler.getCategorieData("Location")

    # training_data, test_data = train_test_split(
    #     category_list, test_size=0.3, random_state=42, shuffle=True
    # )

    # classifier = CategoryClassifier(training_data)

    # modelName = "TestModel"
    # modelCreator = "Giovanni Triulzi"
    # mlPrinciple = "Naive Bayes"
    # refrences = {
    #     "Wikipedia": "https://en.wikipedia.org/wiki/Naive_Bayes_classifier",
    #     "Scikit-learn": "https://scikit-learn.org/stable/modules/naive_bayes.html",
    # }
    # algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naive Bayse.
    # This approach was chosen as it is easy to implement and is computational very efficient.
    # The first step in the classification pipeline is removing all stop words for example 'i', 'me', etc.
    # A list of English stop words is provided by the nltk module. Next the sentence is passed through a stemmer and a lemmatizer.
    # Stemming just removes or stems the last few characters of a word, often leading to incorrect meanings and spelling.
    # Lemmatization considers the context and converts the word to its meaningful base form, which is called Lemma.
    # This is done with the SnowBallStemmer and WordNetLemmatizer class from the nltk module.
    # The final preprocessing step is to vectorize the sentence. For this the Tf-idf vectorizer from sklearn is used.
    # If a Tf-idf vectorizer the sentences don't have to be tokenized.
    # The sentence is now represented in a numerical feature vector which now can be passed to the Naive Bayes classifier."""
    # graphicPath = ""
    # graphicDescription = ""

    # list_of_test_results = []
    # for sent in test_data:
    #     # print(len(sent[0]))
    #     list_of_test_results.append([classifier.classify(sent[0]), sent[1]])
    # print(list_of_test_results)

    # myModelReport = ModelReport(
    #     modelName,
    #     modelCreator,
    #     mlPrinciple,
    #     refrences,
    #     algorithemDescription,
    #     graphicPath,
    #     graphicDescription,
    # )
    # myModelReport.addTrainingSet(training_data)
    # myModelReport.addTestResults(list_of_test_results)
    # myModelReport.createRaport()
    pass

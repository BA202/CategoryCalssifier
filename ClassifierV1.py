import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# for text preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class CategoryClassifier:
    def __init__(self, training_data):
        cleaned_training_data = __preprocess_sentences(training_data)
        category_dict = {
            "Location": 1,
            "Room": 2,
            "Food": 3,
            "Staff": 4,
            "ReasonForStay": 5,
            "GeneralUtility": 6,
            "HotelOrganisation": 7,
            "Unknown": 8,
        }

        category_list = []
        for sent in cleaned_training_data:
            sent_list = [sent[0], category_dict[sent[1]]]
            category_list.append(sent_list)

        df_train = pd.DataFrame(category_list, columns=["sentence", "category"])
        X_train = df_train["sentence"]
        y_train = df_train["category"]

        self.__tfidf_vec = TfidfVectorizer(use_idf=True)
        X_train_vec_tfidf = self.__tfidf_vec.transform(X_train)

        self.__nb_tfidf = MultinomialNB()
        self.__nb_tfidf.fit(X_train_vec_tfidf, y_train)

    def __preprocess_sentences(self, data):
        sent_list = []
        for sentence in data:
            sent_list.append(sentence[0])

        for i in range(len(sent_list)):
            sent_list[i] = sent_list[i].lower()
            sent_list[i] = sent_list[i].replace(".", " ").replace(",", " ")

        list_of_sentence = []
        for sentence in sent_list:
            no_stopwords = []
            for word in sentence.split(" "):
                if word not in stopwords.words("english"):
                    no_stopwords.append(word)
            list_of_sentence.append(" ".join(no_stopwords))

        snow = SnowballStemmer("english")
        list_of_stemmed_sentences = []
        for sentence in list_of_sentence:
            stemmed_sent = []
            for word in word_tokenize(sentence):
                s = snow.stem(word)
                stemmed_sent.append(s)
            list_of_stemmed_sentences.append(" ".join(stemmed_sent))

        wordnet_lemmatizer = WordNetLemmatizer()
        list_of_lemmatized_sentences = []
        for sentence in list_of_lemmatized_sentences:
            lemmatized_sent = []
            for word in sentence.split(" "):
                lemma = wordnet_lemmatizer.lemmatize(word)
                lemmatized_sent.append(lemma)
            list_of_lemmatized_sentences.append(" ".join(lemmatized_sent))

        cleaned_data = []
        for sentence in list_of_lemmatized_sentences:
            for category in data:
                final_list = [sentence, category[1]]
                cleaned_data.append(final_list)

        return cleaned_data

    def classify(self, test_data):
        cleaned_test_data = self.__preprocess_sentences(test_data)

        df_test = pd.DataFrame(cleaned_test_data, columns=["sentence"])
        X_test = df_test["sentence"]

        X_test_vec_tfidf = self.__tfidf_vec.transform(X_test)
        y_predict_test = self.__nb_tfidf.predict(X_test_vec_tfidf)
        y_proba_test = self.__nb_tfidf.predict_proba(X_test_vec_tfidf)[:, 1]

        df_test["predicted_category"] = y_predict_test
        df_test["confidence"] = y_proba_test

        return df_test.values.tolist()


if __name__ == "__main__":
    from DataHandler.DataHandler import DataHandler
    from sklearn.model_selection import train_test_split

    my_data_handler = DataHandler()
    category_list = my_data_handler.getCategorieData("Location")
    for sent in category_list[1:10]:
        print(f"{sent[0]:100}|{sent[1]}")


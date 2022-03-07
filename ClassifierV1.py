import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class CategoryClassifier:
    def __init__(self, trainin_data):
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
        for sent in trainin_data:
            sent_list = [sent[0], category_dict[sent[1]]]
            category_list.append(sent_list)

        df_train = pd.DataFrame(category_list, columns=["sentence", "category"])
        X_train = df_train["sentence"]
        y_train = df_train["category"]

        self.__tfidf_vec = TfidfVectorizer(use_idf=True)
        X_train_vec_tfidf = self.__tfidf_vec.transform(X_train)

        self.__nb_tfidf = MultinomialNB()
        self.__nb_tfidf.fit(X_train_vec_tfidf, y_train)

    def classify(self, test_data):
        df_test = pd.DataFrame(test_data, columns=["sentence"])
        X_test = df_test["sentence"]

        X_test_vec_tfidf = self.__tfidf_vec.transform(X_test)
        y_predict_test = self.__nb_tfidf.predict(X_test_vec_tfidf)
        y_proba_test = self.__nb_tfidf.predict_proba(X_test_vec_tfidf)[:, 1]

        df_test["predicted_category"] = y_predict_test
        df_test["confidence"] = y_proba_test

        return df_test.values.tolist()

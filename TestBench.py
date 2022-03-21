import unittest

# from ScoreClassifierV1 import ScoreClassifierV1
# from ScoreClassifierV15 import ScoreClassifierV15
from CategoryClassifierV15_LDA import CategoryClassifier
from DataHandler.DataHandler import DataHandler
from ModelReport.ModelReport import ModelReport

import random
from time import process_time


class TestbenchClassifier(unittest.TestCase):
    # def test_performanceOfScoreClassifierV15(self):
    #     kfolds = 10

    #     modelName = "ScoreClassificationV15"
    #     modelCreator = "Tobias Rothlin"
    #     mlPrinciple = "Vader Sentiment Analysis"
    #     refrences = {
    #         "Vader overview": "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
    #         "Vader github": "https://github.com/cjhutto/vaderSentiment",
    #     }
    #     algorithemDescription = """Vader (Valence Aware Dictionary for sEntiment Reasoning) is a pre trained model used for sentiment analysis. Vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. The backbone of Vader is a dictionary that maps lexical features to emotion intensities (sentiment score). To receive the sentiment score of a sentence the intensities of each word are added. For example, words like ‘love’, ‘enjoy’ indicating a positive sentiment. Vader is smart enough to understand basic context like ‘did not love’ as negative. Further it has a basic understanding of capitalization and punctuation to emphasis tone.  Due to this any preprocessing steps should not be done. """
    #     graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg_Vader.png"
    #     graphicDescription = "Classification Pipeline"
    #     dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
    #     seed = process_time()

    #     scoreDataHander = DataHandler(
    #         "/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.2"
    #     )
    #     scoreData = scoreDataHander.getScoreData()
    #     random.seed(seed)
    #     random.shuffle(scoreData)

    #     modelPerformanceReport = ModelReport(
    #         modelName,
    #         modelCreator,
    #         mlPrinciple,
    #         refrences,
    #         algorithemDescription,
    #         graphicPath,
    #         graphicDescription,
    #         dataSet,
    #         str(seed),
    #     )

    #     for k in range(kfolds):
    #         testDataStart = int(k * len(scoreData) / kfolds)
    #         testDataEnd = int(k * len(scoreData) / kfolds) + int(
    #             len(scoreData) / kfolds
    #         )
    #         testData = scoreData[testDataStart:testDataEnd]
    #         trainingData = []
    #         for element, i in zip(scoreData, range(len(scoreData))):
    #             if not (i >= testDataStart and i < testDataEnd):
    #                 trainingData.append(element)

    #         print(
    #             f"{k}-training({len(trainingData)}/{(len(trainingData)/(len(trainingData)+len(testData)))*100:.2f}%):Test({len(testData)}/{(len(testData)/(len(trainingData)+len(testData)))*100:.2f}%) Split completed"
    #         )

    #         modelPerformanceReport.addTrainingSet(trainingData)
    #         print(f"{k}-added training split to performance raport")
    #         myScoreClassifier = ScoreClassifierV15(trainingData)
    #         print(f"{k}-model has been trained with training set")
    #         testResults = []

    #         for testCase in testData:
    #             testResults.append(
    #                 [testCase[1], myScoreClassifier.classify(testCase[0])]
    #             )
    #         print(f"{k}-model has been tested\n\n")
    #         modelPerformanceReport.addTestResults(testResults)

    #     print(f"creating the model raport")
    #     modelPerformanceReport.createRaport("ScoreClassifierV15_Vader")

    #     self.assertEqual(True, True)

    # def test_performanceOfScoreClassifierV1(self):
    #     kfolds = 10

    #     modelName = "ScoreClassificationV1"
    #     modelCreator = "Tobias Rothlin"
    #     mlPrinciple = "Multinomial Naive Bayes"
    #     refrences = {
    #         "NultinomialNB Explained": "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
    #         "Stanford NLP Course": "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
    #         "Stanford NLP Lecture": "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
    #         "Engilsh Stopwords": "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm",
    #     }
    #     algorithemDescription = """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """
    #     graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png"
    #     graphicDescription = "Classification Pipeline"
    #     dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
    #     seed = process_time()

    #     scoreDataHander = DataHandler(
    #         "/Users/tobiasrothlin/Documents/BachelorArbeit/DataSets/ClassifiedDataSetV1.2"
    #     )
    #     scoreData = scoreDataHander.getScoreData()
    #     random.seed(seed)
    #     random.shuffle(scoreData)

    #     modelPerformanceReport = ModelReport(
    #         modelName,
    #         modelCreator,
    #         mlPrinciple,
    #         refrences,
    #         algorithemDescription,
    #         graphicPath,
    #         graphicDescription,
    #         dataSet,
    #         str(seed),
    #     )

    #     for k in range(kfolds):
    #         testDataStart = int(k * len(scoreData) / kfolds)
    #         testDataEnd = int(k * len(scoreData) / kfolds) + int(
    #             len(scoreData) / kfolds
    #         )
    #         testData = scoreData[testDataStart:testDataEnd]
    #         trainingData = []
    #         for element, i in zip(scoreData, range(len(scoreData))):
    #             if not (i >= testDataStart and i < testDataEnd):
    #                 trainingData.append(element)

    #         print(
    #             f"{k}-training({len(trainingData)}/{(len(trainingData)/(len(trainingData)+len(testData)))*100:.2f}%):Test({len(testData)}/{(len(testData)/(len(trainingData)+len(testData)))*100:.2f}%) Split completed"
    #         )

    #         modelPerformanceReport.addTrainingSet(trainingData)
    #         print(f"{k}-added training split to performance raport")
    #         myScoreClassifier = ScoreClassifierV1(trainingData)
    #         print(f"{k}-model has been trained with training set")
    #         testResults = []

    #         for testCase in testData:
    #             testResults.append(
    #                 [testCase[1], myScoreClassifier.classify(testCase[0])]
    #             )
    #         print(f"{k}-model has been tested\n\n")
    #         modelPerformanceReport.addTestResults(testResults)

    #     print(f"creating the model raport")
    #     modelPerformanceReport.createRaport("ScoreClassifierV1_NaiveBayes")

    #     self.assertEqual(True, True)

    def test_performanceOfLocationClassifierV2(self):
        kfolds = 10

        modelName = "LocationClassificationV2"
        modelCreator = "Giovanni Triulzi"
        mlPrinciple = "Linear Discriminant Analysis"
        refrences = {
            "LDA Doc.": "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html",
            "Stanford NLP Course": "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
            "Stanford NLP Lecture": "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
            "Engilsh Stopwords": "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm",
        }
        algorithemDescription = """The learning algorithm used in this classification is Linear Discriminant Analysis.
        This approach was chosen as it is easy to implement and is computational very efficient.
        The first step in the classification pipeline is removing all stop words for example 'i', 'me', etc.
        A list of English stop words is provided by the nltk module. Next the sentence is passed through a stemmer and a lemmatizer.
        Stemming just removes or stems the last few characters of a word, often leading to incorrect meanings and spelling.
        Lemmatization considers the context and converts the word to its meaningful base form, which is called Lemma.
        This is done with the SnowBallStemmer and WordNetLemmatizer class from the nltk module.
        The final preprocessing step is to vectorize the sentence. For this the Tf-idf vectorizer from sklearn is used.
        If a Tf-idf vectorizer is used the sentences don't have to be tokenized.
        The sentence is now represented in a numerical feature vector which now can be passed to the LDA classifier."""
        graphicPath = ""
        graphicDescription = ""
        dataSet = f"ClassifiedDataSetV1.2 with {kfolds} folds cross validation"
        seed = process_time()

        scoreDataHander = DataHandler()

        scoreData = scoreDataHander.getCategorieData("Location")
        random.seed(seed)
        random.shuffle(scoreData)

        modelPerformanceReport = ModelReport(
            modelName,
            modelCreator,
            mlPrinciple,
            refrences,
            algorithemDescription,
            graphicPath,
            graphicDescription,
            dataSet,
            str(seed),
        )

        for k in range(kfolds):
            testDataStart = int(k * len(scoreData) / kfolds)
            testDataEnd = int(k * len(scoreData) / kfolds) + int(
                len(scoreData) / kfolds
            )
            testData = scoreData[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(scoreData, range(len(scoreData))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData)/(len(trainingData)+len(testData)))*100:.2f}%):Test({len(testData)}/{(len(testData)/(len(trainingData)+len(testData)))*100:.2f}%) Split completed"
            )

            modelPerformanceReport.addTrainingSet(trainingData)
            print(f"{k}-added training split to performance raport")
            myScoreClassifier = CategoryClassifier(trainingData)
            print(f"{k}-model has been trained with training set")
            testResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])]
                )
            print(f"{k}-model has been tested\n\n")
            modelPerformanceReport.addTestResults(testResults)

        print(f"creating the model raport")
        modelPerformanceReport.createRaport("LocationClassifierV2_LDA")

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()

import re
import string
import os.path
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.base import clone as cloneModel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score, recall_score

import spacy # import it_core_news_sm
from spacy.matcher import Matcher

class Model:
    def __init__(self, lang, logLevel='none'):
        self.lang = lang
        if self.lang == 'it':
            self.nlp = spacy.load('it_core_news_sm')
            stop_w = stopwords.words('italian')
            # Extra stop words detected by TfidfVectorizer
            stop_w_extra = ['aglio', 'avere', 'dare', 'essere', 'facciata', 'fare', 'fossa', 'fosso', 'neo', 'stare', 'stesso', 'torre']
        elif self.lang == 'en':
            self.nlp = spacy.load('en_core_web_sm')
            stop_w = stopwords.words('english')
            stop_w_extra = []
            
        self.stop_words = stop_w + stop_w_extra
        self.logLevel = logLevel
        
    def log(self, obj):
        if self.logLevel == 'debug':
            print(obj)
        pass

    def printAccuracyF1(self, test, pred):
        print(f"Accuracy: {accuracy_score(test, pred)}")
        print(f"F1 weighted average: {f1_score(test, pred, average='weighted')}")
        pass

    def lemmatizeDF(self, data, fileName, class_field=False):
        punctuation_regexp = re.compile("[\.,:;\!\?\|\(\)'<>\-`]")

        # Since the operation is expensive, if lemmatized data was already dumped reads from it.
        if os.path.exists(fileName):
            return pd.read_csv(fileName, index_col="Id")
        new_df = []
        for i, row in data.iterrows():
            text = punctuation_regexp.sub(' ', row["text"])
            # Lemmatize document
            doc = self.nlp(text)
            # Creates a list of lemmas
            lemmas = [word.lemma_ for word in doc]

            if class_field:
                new_df.append([" ".join(lemmas), row["class"]])
            else:
                new_df.append([" ".join(lemmas)])

        if class_field:
            new_df = pd.DataFrame(new_df, columns=["text", "class"])
        else:
            new_df = pd.DataFrame(new_df, columns=["text"])

        # Since the operation is expensive, dumps data to a file
        new_df.to_csv(fileName, index_label="Id")
        return new_df
    
    def gridsearch(self, X, y, unbalanceFactor=1):
        param_search = {
            "penalty": ['l2'], # ['l1', 'l2'],
            "loss": ['squared_hinge', 'hinge'], # ['squared_hinge', 'hinge'],
            "C": [0.8, 1, 1.2, 1.4, 1.6, 1.8],
            #"dual": [True, False],
            "fit_intercept": [True, False],
            "class_weight": [{'pos': unbalanceFactor, 'neg':1}, {'pos':1, 'neg': unbalanceFactor}, 'balanced']
        }
        scorer = make_scorer(f1_score, average='weighted')

        self.log('GridSearching')
        grid = GridSearchCV(LinearSVC(), param_search, scoring=scorer)

        # Find best configuration on full development dataset
        grid.fit(X, y)
        return grid.best_estimator_

    def trainModel(self, data, lemmatized_file="data_lemmatized.csv", gridsearch=True):
        # REALLY, REALLY expensive. It lemmatizes the whole dataset and dumps it to a csv file.
        # If it find the file already stored, reads the lemmatized dataset from it.
        self.log('Lemmatizing')
        lemmatizedData = self.lemmatizeDF(data, lemmatized_file, class_field=True)
        self.log('TFIDF')
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, max_df=0.7, min_df=1, ngram_range=(1,2))
        X_vect = self.vectorizer.fit_transform(lemmatizedData["text"])

        X = X_vect
        y = lemmatizedData['class']
        unbalanceFactor = np.max(y.value_counts())/np.min(y.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

        if gridsearch:
            estimator = self.gridsearch(X, y, unbalanceFactor=unbalanceFactor)
        else:
            estimator = LinearSVC()
            estimator.fit(X,y)
                
        self.estimator = estimator
        classifier = cloneModel(estimator)
        self.log(classifier)

        self.log('Computing F1')
        # Train on training sub-dataset
        classifier.fit(X_train, y_train)
        # Predict on test sub-dataset
        y_pred = classifier.predict(X_test)

        # Compute evaluation metrics for sub-dataset predictions
        self.printAccuracyF1(y_test, y_pred)
        return self.estimator
    
    def predict(self, data):
        if isinstance(data, str):
            data = np.array([data])
        data_vect = self.vectorizer.transform(data)
        return self.estimator.predict(data_vect)
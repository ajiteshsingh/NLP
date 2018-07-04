import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf



documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]



open_file = open("pickled_algos/classifier.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LogisticRegression.pickle", "rb")
LogisticRegression = pickle.load(open_file)
open_file.close()


##open_file = open("pickled_algos/SGDClassifier.pickle", "rb")
##SGDClassifier = pickle.load(open_file)
##open_file.close()


open_file = open("pickled_algos/LinearSVC.pickle", "rb")
LinearSVC = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/NuSVC.pickle", "rb")
NuSVC = pickle.load(open_file)
open_file.close()


voted_classifier = VoteClassifier(classifier,
                                  LinearSVC,
                                  LogisticRegression,
                                  BernoulliNB_classifier,
                                  MNB_classifier)


def sentiment(text):
    feats =find_features(text)

    return voted_classifier.classify(feats)







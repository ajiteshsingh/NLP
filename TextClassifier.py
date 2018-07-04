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

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

##random.shuffle(documents)

all_words =[]
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Naive Bayes Algo accuracy percents : ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

##GaussianNB_classifier = SklearnClassifier(GaussianNB())
##GaussianNB_classifier.train(training_set)
##print("GaussianNB_classifier accuracy percent", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression = SklearnClassifier(LogisticRegression())
LogisticRegression.train(training_set)
print("LogisticRegression accuracy percent", (nltk.classify.accuracy(LogisticRegression, testing_set))*100)

SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print("SGDClassifier accuracy percent", (nltk.classify.accuracy(SGDClassifier, testing_set))*100)

##SVC = SklearnClassifier(SVC())
##SVC.train(training_set)
##print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(SVC, testing_set))*100)

LinearSVC = SklearnClassifier(LinearSVC())
LinearSVC.train(training_set)
print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(LinearSVC, testing_set))*100)

NuSVC = SklearnClassifier(NuSVC())
NuSVC.train(training_set)
print("NuSVC accuracy percent", (nltk.classify.accuracy(NuSVC, testing_set))*100)


voted_classifier = VoteClassifier(classifier,
                                  NuSVC,
                                  LinearSVC,
                                  SGDClassifier,
                                  LogisticRegression,
                                  BernoulliNB_classifier,
                                  MNB_classifier)

print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0])*100)








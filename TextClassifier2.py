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

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

all_words = []
documents = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

 
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(documents, save_word_features)
save_word_features.close()

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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy percents : ", (nltk.classify.accuracy(classifier, testing_set))*100)


save_classifier = open("pickled_algos/classifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_MNB_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_MNB_classifier)
save_MNB_classifier.close()

##GaussianNB_classifier = SklearnClassifier(GaussianNB())
##GaussianNB_classifier.train(training_set)
##print("GaussianNB_classifier accuracy percent", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_BernoulliNB_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_BernoulliNB_classifier)
save_BernoulliNB_classifier.close()



LogisticRegression = SklearnClassifier(LogisticRegression())
LogisticRegression.train(training_set)
print("LogisticRegression accuracy percent", (nltk.classify.accuracy(LogisticRegression, testing_set))*100)

save_LogisticRegression = open("pickled_algos/LogisticRegression.pickle","wb")
pickle.dump(LogisticRegression, save_LogisticRegression)
save_LogisticRegression.close()

SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print("SGDClassifier accuracy percent", (nltk.classify.accuracy(SGDClassifier, testing_set))*100)

save_SGDClassifier = open("pickled_algos/SGDClassifier.pickle","wb")
pickle.dump(SGDClassifier, save_SGDClassifier)
save_SGDClassifier.close()

##SVC = SklearnClassifier(SVC())
##SVC.train(training_set)
##print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(SVC, testing_set))*100)

LinearSVC = SklearnClassifier(LinearSVC())
LinearSVC.train(training_set)
print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(LinearSVC, testing_set))*100)

save_LinearSVC = open("pickled_algos/LinearSVC.pickle","wb")
pickle.dump(LinearSVC, save_LinearSVC)
save_LinearSVC.close()

NuSVC = SklearnClassifier(NuSVC())
NuSVC.train(training_set)
print("NuSVC accuracy percent", (nltk.classify.accuracy(NuSVC, testing_set))*100)

save_NuSVC = open("pickled_algos/NuSVC.pickle","wb")
pickle.dump(NuSVC, save_NuSVC)
save_NuSVC.close()


voted_classifier = VoteClassifier(classifier,
                                  NuSVC,
                                  LinearSVC,
                                  SGDClassifier,
                                  LogisticRegression,
                                  BernoulliNB_classifier,
                                  MNB_classifier)

print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats =find_features(text)

    return voted_classifier.classify(feats)







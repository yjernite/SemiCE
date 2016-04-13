#~ Reads a dictionary (UMLS), labeled data (mention scopes and labels,
#~ Semeval), and unlabeled data (mention scopes without label, MIMIC) and
#~ outputs hdf5 files to be read by the Torch code.
import os
import h5py
import cPickle as pickle
import numpy as np
from os.path import join as pjoin
from difflib import SequenceMatcher
from subprocess import call

data_dir = os.path.abspath('../Data')
output_dir = os.path.abspath('pre_processed')

## Load dictionary
# The dictionary is provided as a python dict, with labels as keys and
# lists of descriptions as values. For example:
#     dictionary["C0085593"] = ["chill", "chills", "shaking chill", ...]
dictionary_file = pjoin(data_dir, 'UMLS_dict.pk')
dictionary = pickle.load(open(dictionary_file))

## Load labeled data
# Labeled data is given as a list of documents.
# Each document is a a list of sentences.
# Each sentence is a pair containing the tokenized string and a list of
# labeled mentions, for example:
#     ("Patient complains of strong fevers and chills .",
#       [((3, 4), "C0015967"), ((6,), "C0085593")])
labeled_file = pjoin(data_dir, 'semeval_sentences.pk')
labeled_data = pickle.load(open(labeled_file))

## Load unlabeled data
# The unlabeled data has the same format as the labeled data, but
# only the mention spans are given
#     ("Patient complains of strong fevers and chills .",
#       [(3, 4), (6,)])
unlabeled_file = pjoin(data_dir, 'mimic_sentences.pk')
unlabeled_data = pickle.load(open(unlabeled_file))

## Make vocabularies, supports
vocab_counts = {}
mention_counts = {}
for document in labeled_data + unlabeled_data:
    for sentence, mentions in document:
        sentence_split = sentence.split()
        for w in sentence_split:
            vocab_counts[w] = vocab_counts.get(w, 0) + 1
        for mention in mentions:
            if type(mention[0]) == type(0):
                men_text = ' '.join([sentence_split[i] for i in mention])
            else:
                men_text = ' '.join([sentence_split[i] for i in mention[0]])
            mention_counts[men_text] = mention_counts.get(men_text, 0) + 1

dic_vocab_counts = {}
dic_mention_counts = {}
for concept, mentions in dictionary.items():
    for mention in mentions:
        dic_mention_counts[mention] = dic_mention_counts.get(mention, 0) + 1
        for w in mention.split():
            dic_vocab_counts[w] = dic_vocab_counts.get(w, 0) + 1

labels_voc = ['<UNK>', '<S>'] + dictionary.keys()
words_voc = ['<UNK>', '<S>'] + list(set(vocab_counts.keys() + dic_vocab_counts.keys()))
mentions_voc = ['<UNK>', '<S>'] + list(set(mention_counts.keys() + dic_mention_counts.keys()))


# For each mention, we define a smaller set of possible labels, or support.
# There are several ways to do this pre-processing step depending on the
# application domain. Here, we rely on a string distance metric and threshold
# the distance between a mention and the label descriptions given in the
# dictionary.
def str_distance(mention, label):
    return max([SequenceMatcher(None, mention, desc).ratio()
                for desc in dictionary.get(label, [])])


def get_support(mention, threshold=0.5):
    res = [label for label in dictionary if str_distance(mention, label) > threshold]
    return ['<UNK>'] + res

mention_supports = [get_support(mention) for mention in mentions_voc]
mention_supports[0] = ['<UNK>']
mention_supports[1] = ['<S>']

## Make feature representations (neighbours)
word_k = 5
mention_k = 4


# extracts features: left and right words and mentions, label.
# returns a list of documents
def make_features(data):
    features = []
    for document in data:
		doc_features = []
        for sentence, mentions in document:
        sentence_split = sentence.split()
        for mention in mentions:
			feature = {}
			feature['left_words'] = ['<S>'] * word_k
			feature['right_words'] = ['<S>'] * word_k
			feature['left_mentions'] = ['<S>'] * mention_k
			feature['right_mentions'] = ['<S>'] * mention_k
            if type(mention[0]) == type(0):
                men_text = ' '.join([sentence_split[i] for i in mention])
                label = '<UNK>'
                first = mention[0]
                last = mention[-1]
            else:
                men_text = ' '.join([sentence_split[i] for i in mention[0]])
                label = mention[1]
                first = mention[0][0]
                last = mention[0][-1]
            feature['mention'] = men_text
            feature['label'] = label
            for k in range(1, word_k + 1):
				if first - k > 0:
					feature['left_words'][word_k - k] = sentence_split[first - k]
				if last + k < len(sentence_split):
					feature['right_words'][k - 1] = sentence_split[last + k]
			doc_features += [feature]
		for i, feature in enumerate(doc_features):
			for k in range(1, mention_k + 1):
				if i - k > 0:
					feature['left_mentions'][mention_k - k] = doc_features[i - k]['mention']
				if i + k < len(doc_features):
					feature['right_mentions'][k - 1] = doc_features[i + k]['mention']
		features += doc_features[:]
    return features

word_lookup = dict([(w, i) for i, w in enumerate(words_voc)])
mention_lookup = dict([(w, i) for i, w in enumerate(mentions_voc)])
label_lookup = dict([(w, i) for i, w in enumerate(labels_voc)])


def read_features(features):
    res = []
    for w in feature['left_words'] + feature['right_words']:
        res += [word_lookup[w]]
    for men in feature['left_mentions'] + feature['right_mentions']:
        res += [mention_lookup[men]]
    res += [mention_lookup[feature['mention']]]
    res += [label_lookup['label']]
    return res


def make_features_array(features):
    array = np.zeros((len(features), 2 * word_k + 2 * mention_k + 2))
    for i, feature in enumerate(features):
        for j, feat in enumerate(read_feature(feature)):
            array[i, j] = feat
    return array

# data in array format
sup_array = make_features_array(make_features(labeled_data))
unsup_array = make_features_array(make_features(unlabeled_data))

max_sup = max([len(sup) for sup in mention_supports])
support_array = np.zeros((len(mention_supports), max_sup))
for i, men in enumerate(mentions_voc):
    sup = [label_lookup[label] for label in mention_supports[men]]
    for j, label in enumerate(sorted(sup)):
        support_array[i, j] = label

max_words = max([len(mention.split()) for mention in mentions_voc])
mention_to_words_array = np.zeros((len(mention_supports), max_words))
for i, men in enumerate(mentions_voc):
    for j, w in enumerate(men.split()):
        mention_to_words_array[i, j] = word_lookup[w]

## Write to hdf5
call(["mkdir", output_dir])

f = h5py.File(pjoin(output_dir, "features.hdf5"), "w")
dset_sup = f.create_dataset("supervised", data=sup_array)
dset_unsup = f.create_dataset("unsupervised", data=unsup_array)
dset_supports = f.create_dataset("support", data=support_array)
dset_men_to_words = f.create_dataset("men_to_words", data=mention_to_words_array)
f.close()

## Write vocab files
f = open(pjoin(output_dir, "mention_vocab.txt"), "w")
for mention in mentions_voc:
    print >>f, mention

f.close()

f = open(pjoin(output_dir, "label_vocab.txt"), "w")
for label in labels_voc:
    print >>f, label

f.close()

f = open(pjoin(output_dir, "word_vocab.txt"), "w")
for word in words_voc:
    print >>f, word

f.close()

print('All done')


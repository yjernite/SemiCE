#~ Reads a dictionary (UMLS), labeled data (mention scopes and labels,
#~ Semeval), and unlabeled data (mention scopes without label, MIMIC) and
#~ outputs hdf5 files to be read by the Torch code.
import os
import h5py
import cPickle as pickle
import numpy as np
from os.path import join as pjoin
from nltk.tokenize import word_tokenize as split_tokens

data_dir = '../Data'

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

label_voc = ['<UNK>', '<S>'] + dictionary.keys()
words_voc = ['<UNK>', '<S>'] + list(set(vocab_counts.keys() + dic_vocab_counts.keys()))
mentions_voc = ['<UNK>', '<S>'] + list(set(mention_counts.keys() + dic_mention_counts.keys()))

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


# TODO: here
def read_features(features):
    res = []
    for w in feature['left_words'] + feature['right_words']:
        res += [word_lookup[w]]
    for men in feature['left_mentions'] + feature['right_mentions']:
        res += [mention_lookup[men]]
    res += [mention_lookup[feature['mention']]]
    res += [cui_lookup.get(feature['label'], cui_lookup['CUI-less'])]
    return res


def make_features_array(features):
    array = np.zeros((len(features), 2 * word_k + 2 * mention_k + 2))
    for i, feature in enumerate(features):
        for j, feat in enumerate(read_feature(feature)):
            array[i, j] = feat
    return array


# data in array format
(train_array, train_array_full) = make_features_array(train_features, train_features_full)
(dev_array, dev_array_full) = make_features_array(dev_features, dev_features_full)
(unsup_array, unsup_array_full) = make_features_array(unsup_features, unsup_features_full)


## Write to hdf5
# supports
max_sup = max([len(sup) for sup in mention_supports.values()]) + 2
support_array = np.ones((len(mention_supports), max_sup))
for i, men in enumerate(mention_vocab):
    sup = [cui_lookup[cui] for cui in mention_supports.get(men, []) + ['CUI-less'] + ['NONE']]
    for j, cui in enumerate(sorted(sup)):
        support_array[i, j] = cui

# train matches
max_train = max([len(sup) for sup in train_lookup.values()])
train_lookup_array = np.ones((len(mention_vocab), max_train))
for i, men in enumerate(mention_vocab):
    sup = [cui_lookup.get(cui, 1) for cui in train_lookup.get(men, [])]
    for j, cui in enumerate(sorted(sup)):
        train_lookup_array[i, j] = cui

# umls matches
max_umls = max([len(sup) for sup in my_lookup.values()]) + 1
umls_lookup_array = np.ones((len(mention_vocab), max_umls))
for i, men in enumerate(mention_vocab):
    sup = [cui_lookup[cui] for cui in my_lookup.get(men, [])+ ['CUI-less']]
    for j, cui in enumerate(sorted(sup)):
        umls_lookup_array[i, j] = cui


# alternatively, one support file
def candidates(st):
    poss = []
    if st in train_lookup:
        return (train_lookup[st], 'Training')
    elif st in my_lookup:
        for cui in my_lookup[st]:
            if UMLS[cuitoid[cui]][1] == st:
                return ([cui], 'Concept_name')
        return (my_lookup[st] + ['NONE'], 'Lookup')
    else:
        supp = pre_select(st, my_lookup, has_pref) + ['CUI-less'] + ['NONE']
        return (supp, 'Support')

max_all = max(max_sup, max_umls + max_train)
full_support_array = np.ones((len(mention_supports), max_all))

for i, men in enumerate(mention_vocab):
    (poss, reason) = candidates(men)
    sup = [cui_lookup.get(cui, 1) for cui in poss]
    for j, cui in enumerate(sorted(sup)):
        full_support_array[i, j] = cui

print('Made support arrays')

f = h5py.File(save_dir + "/features.hdf5", "w")
dset_train = f.create_dataset("train", data=train_array)
dset_dev = f.create_dataset("dev", data=dev_array)
dset_unsup = f.create_dataset("unsup", data=unsup_array)
dset_train = f.create_dataset("train_full", data=train_array_full)
dset_dev = f.create_dataset("dev_full", data=dev_array_full)
dset_unsup = f.create_dataset("unsup_full", data=unsup_array_full)
f.close()

f = h5py.File(save_dir + "/supports.hdf5", "w")
dset_sup = f.create_dataset("support", data=support_array)
dset_train = f.create_dataset("train_lookup", data=train_lookup_array)
dset_umls = f.create_dataset("umls_lookup", data=umls_lookup_array)
f.close()

f = h5py.File(save_dir + "/full_support.hdf5", "w")
dset_sup = f.create_dataset("full_support", data=full_support_array)
f.close()


print('All done')


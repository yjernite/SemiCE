#~ Reads a dictionary (UMLS), labeled data (mention scopes and labels, Semeval),
#~ and unlabeled data (mention scopes without label) and outputs hdf5 files
#~ to be read by the Torch code.
import h5py
import cPickle as pickle
from os.path import join as pjoin
from nltk.tokenize import word_tokenize as split_tokens

data_dir = '../Data'

## Load dictionary
# The dictionary is provided as a python dict, with labels as keys and
# lists of descriptions as values. For example:
# 	dictionary["C0085593"] = ["chill", "chills", "shaking chill", ...]
dictionary_file = pjoin(data_dir, 'UMLS_dict.pk')
dictionary = pickle.load(open(dictionary_file))

## Load labeled data
# Labeled data is given as a list of documents.
# Each document is a a list of sentences.
# Each sentence is a pair containing the tokenized string and a list of
# labeled mentions, for example:
# 	("Patient complains of strong fevers and chills .",
#  	 [((3, 4), "C0015967"), ((6,), "C0085593")])
labeled_file = pjoin(data_dir, 'semeval_sentences.pk')
labeled_data = pickle.load(open(labeled_file))

## Load unlabeled data
# The unlabeled data has the same format as the labeled data, but
# only the mention spans are given
# 	("Patient complains of strong fevers and chills .",
#  	 [(3, 4), (6,)])
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

## Make feature representations (neighbours)

## Write to hdf5


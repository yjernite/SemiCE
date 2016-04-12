from os.path import join as pjoin
import sys
import cPickle as pickle

git_dir = '/home/jernite/Code/git/ConceptExtraction'
sys.path.append(pjoin(git_dir, 'Detection/'))
from ReadUMLS import *

train_labels_file = pjoin(git_dir, 'Detection/train_concepts1.dat')
dev_labels_file = pjoin(git_dir, 'Detection/dev_concepts1.dat')
train_spans_file = pjoin(git_dir, 'Detection/crfpp_input_train/crfpp_spans_batch_1.txt')
dev_spans_file = pjoin(git_dir, 'Detection/crfpp_input_dev/crfpp_spans_batch_1.txt')
train_text_file = pjoin(git_dir, 'Detection/crfpp_input_train/crfpp_tokenized_batch_1.txt')
dev_text_file = pjoin(git_dir, 'Detection/crfpp_input_dev/crfpp_tokenized_batch_1.txt')

UMLSfile = pjoin(git_dir, 'data/UMLStok.dat')


def read_labels(labels_file):
    labels = {}
    f = open(labels_file)
    for line in f:
        tab = line.strip().split('\t')
        if len(tab) > 1:
            note_id = tab[0].strip()
            pre_st = tab[1].strip()[1:-1]
            spans_list = [(int(sp.split(', ')[0]), int(sp.split(', ')[1]))
                  for sp in pre_st.split(') (')]
            st = tab[2].strip()
            label = tab[3].strip()
            labels[note_id] = labels.get(note_id, []) + \
                                  [[spans_list, st, label]]
    f.close()
    return labels


def read_spans(spans_file, text_file):
    spans = {}
    text = {}
    sentence = []
    sentence_words = []
    note_id = ''
    f = open(text_file)
    text_lines = f.readlines()
    f.close()
    f = open(spans_file)
    i = 0
    for line in f:
        word = text_lines[i].strip()
        i += 1
        if len(line.strip()) == 0:
            if note_id in spans and spans[note_id][-1][0] == sentence[0]:
                sentence = []
                sentence_words = []
            else:
                spans[note_id] = spans.get(note_id, []) + [sentence[:]]
                text[note_id] = text.get(note_id, []) + [sentence_words[:]]
                sentence = []
                sentence_words = []
        else:
            tab = line.strip().split('\t')
            note_id = tab[1]
            span = tuple([int(st.strip()) for st in tab[0].split()])
            sentence += [span]
            sentence_words += [word]
    f.close()
    return spans, text


def pre_select(st, my_lookup, has_pref):
    res = my_lookup.get(st, [])[:]
    for w in st.split():
        if w in has_pref:
            res += has_pref[w]
        if len(w) > 5 and w[:5] in has_pref:
            res += has_pref[w[:5]]
    res = list(set(res))
    return res


# Read data files
train_labels = read_labels(train_labels_file)
train_spans, train_text = read_spans(train_spans_file, train_text_file)
dev_labels = read_labels(dev_labels_file)
dev_spans, dev_text = read_spans(dev_spans_file, dev_text_file)


# Read UMLS
UMLS, lookup, trie, prefix_trie, suffix_trie, \
    spelling_trie, acro_trie = read_umls(UMLSfile)

cuitoid = dict([(co[0], co[2]) for co in UMLS])

candidate_cuis = pickle.load(open('candidate_cuis.pk'))
my_counts = dict([(cui, 1) for cui in candidate_cuis])

# Lookup dictionaries
train_lookup = {}
for note_id, note_labels in train_labels.items():
    for label in note_labels:
        train_lookup[label[1]] = train_lookup.get(label[1], []) + [label[2]]
        my_counts[label[2]] = 1

for st in train_lookup:
    train_lookup[st] = list(set(train_lookup[st]))

lookup_rest = {}
for cui in my_counts:
    if cui in cuitoid:
        for st in UMLS[cuitoid[cui]][4]:
            lookup_rest[st] = lookup_rest.get(st, []) + [cui]

# Define supports
my_lookup = {}
stop_words = ['the', 'of', 'in', 'to', 'nos']
stop_words += ['unspecified', 'disease', 'disorder']
has_pref = {}
for cui in my_counts:
    if cui not in cuitoid:
        print cui
        continue
    add_list = []
    for st in UMLS[cuitoid[cui]][4]:
        my_lookup[st] = my_lookup.get(st, []) + [cui]
        acro = ''.join([w[0] for w in st.split()])
        prefs = [w[:5] for w in st.split()]
        if acro.isalpha():
            add_list += [w for w in [acro] + prefs + st.split() if w.isalpha()]
    for w in list(set(add_list)):
        if w.isalpha():
            has_pref[w] = has_pref.get(w, []) + [cui]

for w in stop_words:
    if w in has_pref:
        del has_pref[w]
    if w[:5] in has_pref:
        del has_pref[w[:5]]

for w in has_pref:
    has_pref[w] = list(set(has_pref[w]))

for men in my_lookup:
    my_lookup[men] = list(set(my_lookup[men]))


print('Made lookups')


# Read MIMIC exact matches
from os.path import join as pjoin
from nltk.tokenize import word_tokenize as split_tokens
import re
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')


def spans_to_str(sp_list, sentences):
    res = []
    for i, j in sp_list:
        this_span = sentences[i][j]
        if len(res) == 0 or this_span[0] > res[-1][1] + 1:
            res += [this_span]
        else:
            res[-1] = (res[-1][0], this_span[1])
    st_res = ','.join([str(sp[0]) + '-' + str(sp[1]) for sp in res])
    return st_res


def read_features(file_name, spans_dic={}, thres=0.5, mention_k=4):
    ct = 0
    f = open(file_name)
    res = []
    note = []
    for line in f:
        mention = line.strip().split('\t')
        if len(mention) > 1 and (mention[3][0] == 'C' or float(mention[4]) > thres):
            ct += 1
            if ct % 500000 == 0:
                print ct
            mention_dic = {}
            mention_dic['crf_score'] = float(mention[4])
            mention_dic['left_words'] = mention[5].split()
            mention_dic['right_words'] = mention[6].split()
            mention_dic['label'] = mention[3]
            if mention_dic['label'] == 'None' or 'Ambiguous' in mention_dic['label']:
                mention_dic['label'] == 'UNK'
            mention_dic['mention'] = mention[2]
            mention_dic['file_name'] = mention[0]
            if len(spans_dic) > 0:
                spans_list = [(int(sp.split(', ')[0]), int(sp.split(', ')[1]))
                              for sp in mention[1].strip()[1:-1].split(') (')]
                mention_dic['span'] = spans_to_str(spans_list, spans_dic[mention[0]])
            note += [mention_dic]
        elif len(mention) <= 1 and len(note) > 0:
            res += [note[:]]
            note = []
    if len(note) > 0:
        res += [note[:]]
    for note in res:
        pre_mentions = ['<M>'] * mention_k
        for i, mention in enumerate(note):
            mention['left_mentions'] = pre_mentions[:]
            pre_mentions = pre_mentions[1:] + [mention['mention']]
            if i >= mention_k:
                note[i - mention_k]['right_mentions'] = pre_mentions[:]
        for j in range(mention_k):
            if j >= mention_k - len(note):
                pre_mentions = pre_mentions[1:] + ['</M>']
                note[len(note) - mention_k + j]['right_mentions'] = pre_mentions[:]
    f.close()
    return res


word_k = 5
mention_k = 4

mimic_features = {}
for i in range(33):
    num = "%02d" % (i,)
    mimic_features[num] = []
    print 'file', num
    mentions_file = pjoin(git_dir, 'Detection/MIMICcrf/%s/mimic_%s_concepts.dat' % (num, num))
    feats = read_features(mentions_file)
    mimic_features[num] = feats


unsup_features = []
unsup_features_full = []
for num, notes in mimic_features.items():
    print num
    for note in notes:
        unsup_features += [mention for mention in note if mention['label'][0] == 'C']
        unsup_features_full += [mention for mention in note]


# make train and dev features. TODO: get IDs
train_features_file = pjoin(git_dir, 'Detection/MIMICcrf/train_concepts.dat')
train_features = []
for note in read_features(train_features_file, train_spans):
    train_features += [mention for mention in note if mention['label'][0] == 'C']

train_features_full = []
for note in read_features(train_features_file, train_spans, 0):
    train_features_full += [mention for mention in note]

dev_features_file = pjoin(git_dir, 'Detection/MIMICcrf/dev_concepts.dat')
dev_features = []
for note in read_features(dev_features_file, dev_spans):
    dev_features += [mention for mention in note if mention['label'][0] == 'C']

dev_features_full = []
for note in read_features(dev_features_file, dev_spans, 0):
    dev_features_full += [mention for mention in note]


print('Made features')

# Make vocabularies
word_vocab = []
word_lookup = {}
mention_vocab = []
mention_lookup = {}
cui_vocab = []
cui_lookup = {}

word_counts = {}
mention_counts = {}

ct = 0
for feature in train_features_full + dev_features_full + unsup_features_full:
    ct += 1
    if ct % 100000 == 0:
        print 100 * float(ct) / 3.25e7
    for men in [feature['mention']] + feature['left_mentions'] + \
               feature['right_mentions']:
        mention_counts[men] = mention_counts.get(men, 0) + feature['crf_score']
        if men not in mention_lookup:
            mention_vocab += [men]
            mention_lookup[men] = len(mention_vocab)  # 1-indiced
    for w in feature['left_words'] + feature['right_words']:
        word_counts[w] = word_counts.get(w, 0) + feature['crf_score']
        if w not in word_lookup:
            word_vocab += [w]
            word_lookup[w] = len(word_vocab)


for cui in ['CUI-less'] + ['NONE'] + ['UNK'] + my_counts.keys():
    if not cui in cui_lookup:
        cui_vocab += [cui]
        cui_lookup[cui] = len(cui_vocab)

##### HERE
# filter vocab
word_vocab = [w for w, ct in word_counts.items() if ct > 9] + ['<UNK>']
for w in word_lookup:
    word_lookup[w] = len(word_vocab)

for i, w in enumerate(word_vocab):
    word_lookup[w] = i + 1

# filter mentions
mention_vocab = [w for w, ct in mention_counts.items() if ct > 19] + ['<UNK>']
for w in mention_lookup:
    mention_lookup[w] = len(mention_vocab)

for i, w in enumerate(mention_vocab):
    mention_lookup[w] = i + 1

# Make supports
mention_supports = {}
for men in mention_vocab:
    mention_supports[men] = pre_select(men, my_lookup, has_pref)



print('Made vocabs and supports')

# Save in hdf5
import numpy as np
import h5py

word_k = 5
mention_k = 4

def read_feature(feature):
    res = []
    for w in feature['left_words'] + feature['right_words']:
        res += [word_lookup[w]]
    for men in feature['left_mentions'] + feature['right_mentions']:
        res += [mention_lookup[men]]
    res += [mention_lookup[feature['mention']]]
    res += [cui_lookup.get(feature['label'], cui_lookup['CUI-less'])]
    return res


def make_features_array(features, features_full):
    array = np.zeros((len(features), 2 * word_k + 2 * mention_k + 2))
    for i, feature in enumerate(features):
        for j, feat in enumerate(read_feature(feature)):
            array[i, j] = feat
    array_full = np.zeros((len(features_full), 2 * word_k + 2 * mention_k + 2))
    for i, feature in enumerate(features_full):
        for j, feat in enumerate(read_feature(feature)):
            array_full[i, j] = feat
    return (array, array_full)


# data in array format
(train_array, train_array_full) = make_features_array(train_features, train_features_full)
(dev_array, dev_array_full) = make_features_array(dev_features, dev_features_full)
(unsup_array, unsup_array_full) = make_features_array(unsup_features, unsup_features_full)

print('Made feature arrays')

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

# Save everything
import os
import cPickle as pickle

save_dir = pjoin(git_dir, 'Identification/feature_files_full')

os.system('mkdir ' + save_dir)
pickle.dump((word_vocab, word_lookup, mention_vocab, mention_lookup,
             cui_vocab, cui_lookup), open(save_dir + '/dictionaries.pk', 'wb'))


pickle.dump((train_features, dev_features, train_features_full,
             dev_features_full, unsup_features_full[:50000]),
            open(save_dir + '/features.pk', 'wb'))

import h5py
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

# Process words2vec output into hd5f

cui_rep_m_file = 'pre_trained_vecs/cui_vecs_m.dat'
cui_rep_w_file = 'pre_trained_vecs/cui_vecs_w.dat'

word_rep_file = 'pre_trained_vecs/word_vecs.dat'
mention_rep_file = 'pre_trained_vecs/mention_vecs.dat'

cui_reps_m = {}
f = open(cui_rep_m_file)
for line in f:
    tab = line.strip().split()
    cui_reps_m[tab[0]] = np.array(map(float, tab[1:]))

f.close()

cui_reps_w = {}
f = open(cui_rep_w_file)
for line in f:
    tab = line.strip().split()
    cui_reps_w[tab[0]] = np.array(map(float, tab[1:]))

f.close()

cui_rep_dim = len(cui_reps_w.values()[0])
cui_reps_vec = np.zeros((len(cui_vocab), cui_rep_dim))
for i, cui in enumerate(cui_vocab):
    cui_reps_vec[i] += cui_reps_m.get(cui, 0.1 * (np.random.rand(cui_rep_dim) - 0.5))
    cui_reps_vec[i] += cui_reps_w.get(cui, 0.1 * (np.random.rand(cui_rep_dim) - 0.5))


word_reps = {}
f = open(word_rep_file)
for line in f:
    tab = line.strip().split()
    word_reps[tab[0]] = np.array(map(float, tab[1:]))

f.close()

word_rep_dim = len(word_reps.values()[0])
word_reps_vec = np.zeros((len(word_vocab), word_rep_dim))
for i, word in enumerate(word_vocab):
    word_reps_vec[i] = word_reps.get(word, 0.2 * (np.random.rand(word_rep_dim) - 0.5))


mention_reps = {}
f = open(mention_rep_file)
for line in f:
    tab = line.strip().split()
    mention_reps[tab[0]] = np.array(map(float, tab[1:]))

f.close()

mention_rep_dim = len(mention_reps.values()[0])
mention_reps_vec = np.zeros((len(mention_vocab), mention_rep_dim))
for i, men in enumerate(mention_vocab):
    mention = '__'.join(men.split())
    mention_reps_vec[i] = mention_reps.get(mention, 0.2 * (np.random.rand(mention_rep_dim) - 0.5))


f = h5py.File(save_dir + "/pre_trained_100.hdf5", "w")
dset_words = f.create_dataset("words", data=word_reps_vec)
dset_mentions = f.create_dataset("mentions", data=mention_reps_vec)
dset_cuis = f.create_dataset("cuis", data=cui_reps_vec)
f.close()

print('All done')

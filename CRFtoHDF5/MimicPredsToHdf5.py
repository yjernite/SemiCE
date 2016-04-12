############### INCOMPLETE!
### label unsup data given predictions

from os.path import join as pjoin
import sys
import cPickle as pickle
import numpy as np

from EvaluatePredictions import *

git_dir = '/home/jernite/Code/git/ConceptExtraction'

##########################
###  make MIMIC features
##########################

word_k = 5
mention_k = 4


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

mimic_features = {}
for i in range(33):
    num = "%02d" % (i,)
    mimic_features[num] = []
    print 'file', num
    mentions_file = pjoin(git_dir, 'Detection/MIMICcrf/%s/mimic_%s_concepts.dat' % (num, num))
    feats = read_features(mentions_file)
    mimic_features[num] = feats


unsup_features_full = []
for num, notes in mimic_features.items():
    print num
    for note in notes:
        unsup_features_full += [mention for mention in note]


save_dir = pjoin(git_dir, 'Identification/feature_files_full')

(word_vocab, word_lookup, mention_vocab, mention_lookup, cui_vocab, cui_lookup) = pickle.load(open(save_dir + '/dictionaries.pk'))


def read_feature(feature, preds):
    res = []
    for w in feature['left_words'] + feature['right_words']:
        res += [word_lookup[w]]
    for men in feature['left_mentions'] + feature['right_mentions']:
        res += [mention_lookup[men]]
    res += [mention_lookup[feature['mention']]]
    #TODO: if not known, use preds to predict
    res += [cui_lookup.get(feature['label'], cui_lookup['CUI-less'])]
    return res


def make_features_array(features_full, preds_file):
	array_full = np.zeros((len(features_full), 2 * word_k + 2 * mention_k + 2))
	preds_f = h5py.File(preds_file,'r')
    su_preds = preds_f['predictions']
	for i, feature in enumerate(features_full):
		for j, feat in enumerate(read_feature(feature, su_preds[i])):
			array_full[i, j] = feat
	return (array, array_full)


(unsup_array, unsup_array_full) = make_features_array(unsup_features, unsup_features_full)


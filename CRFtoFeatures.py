import sys
from os.path import join as pjoin

git_dir = '/home/jernite/Code/git/ConceptExtraction'
sys.path.append(pjoin(git_dir, 'Detection/'))

from ReadUMLS import *
from ReadCRFOutput import *

# Process CRF output

git_dir = '/home/jernite/Code/git/ConceptExtraction'
UMLSfile = pjoin(git_dir, 'data/UMLStok.dat')

UMLS, lookup, trie, prefix_trie, suffix_trie, \
    spelling_trie, acro_trie = read_umls(UMLSfile)

# First cut of UMLS: largest connected component of good_types
# TODO: all good_types
UMLStreefile = pjoin(git_dir, '../ML-tools/UMLS/UMLStree.dat')
cuitoid = dict([(co[0], co[2]) for co in UMLS])
CUIrestlist = []
f = open(UMLStreefile)
for line in f:
    tab=line.split('\t')
    if len(tab) > 1:
        CUIrestlist += [tab[3].strip()]

f.close()
lookup_rest = {}
for cui in CUIrestlist:
    concept = UMLS[cuitoid[cui]]
    for st in concept[4]:
        lookup_rest[st] = lookup_rest.get(st, []) + [cui]


def match_umls(st):
    poss = lookup_rest.get(st.lower(), [])
    if len(poss) == 0:
        return 'None'
    elif len(poss) == 1:
        return poss[0]
    else:
        for cui in poss:
            if UMLS[cuitoid[cui]] == st.lower():
                return cui
        return '_'.join(['Ambiguous'] + poss)


# For UMLS, label is exact UMLS match
def treat_merged_sentence_umls(me, num=0, thr=0.25):
    res = []
    sen = me[0]
    text = sen[0].lower().split()
    span_list = me[1]
    identified = sorted([(a, b) for a, b in sen[2] if b>thr], key=lambda x:x[0])
    for sp, score in identified:
        st = ' '.join([text[i] for i in sp])
        label = match_umls(st)
        file_num = span_list[0][2]
        span = [(num, i) for i in sp]
        padded = ['<S>'] * 5 + text + ['</S>'] * 5
        start = sp[0]
        end = sp[-1]
        left_words = ' '.join(padded[start: start+5])
        right_words = ' '.join(padded[end + 6:end + 11])
        res += [[file_num, ' '.join(map(str, span)), st, label, str(score),
                 left_words[:], right_words[:]]]
    return res


# TODO: manage folders better
MIMIC_crf_dir = pjoin(git_dir, 'Detection/MIMICcrf')
# process file
for j in range(33):
    print "batch", j
    batch = "%02d" % (j,)
    results_file = '%s/%s/mimic_%s_res_c1_l2_v2.dat' % (MIMIC_crf_dir, batch, batch)
    spans_file = '%s/%s/crfpp_spans_batch_1.txt' % (MIMIC_crf_dir, batch)
    sentences = treat_sentences(results_file)
    print "read sentences"
    spans = treat_spans(spans_file)
    merged = merge(sentences, spans)
    print "merged"
    output_file = '%s/%s/mimic_%s_concepts.dat' % (MIMIC_crf_dir, batch, batch)
    f = open(output_file, 'w')
    file_name = ''
    num = 0
    for i, me in enumerate(merged):
        new_file_name = me[1][0][2]
        if new_file_name == file_name:
            num += 1
        else:
            file_name = new_file_name
            num = 0
            print >>f, ''
        rlist = treat_merged_sentence_umls(me, num)
        for ls in rlist:
            print >>f, '\t'.join(ls)
    f.close()


############################
### Get similar format for train and dev sets, with labels
############################

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
            sentence_num = spans_list[0][0]
            word_nums = [sp[1] for sp in spans_list]
            st = tab[2].strip()
            label = tab[3].strip()
            labels[note_id] = labels.get(note_id, {})
            span = (sentence_num, tuple(word_nums))
            labels[note_id][span] = (st, label)
    f.close()
    return labels


def treat_merged_sentence(me, labels_dict, num=0, thr=0.25):
    res = []
    sen = me[0]
    text = sen[0].lower().split()
    span_list = me[1]
    identified = sorted([(a, b) for a, b in sen[2] if b>thr or a in sen[1]],
                        key=lambda x:x[0])
    for sp, score in identified:
        st = ' '.join([text[i] for i in sp])
        label = labels_dict.get((num, sp), ('', 'NONE'))[1]
        file_num = span_list[0][2]
        span = [(num, i) for i in sp]
        padded = ['<S>'] * 5 + text + ['</S>'] * 5
        start = sp[0]
        end = sp[-1]
        left_words = ' '.join(padded[start: start+5])
        right_words = ' '.join(padded[end + 6:end + 11])
        res += [[file_num, ' '.join(map(str, span)), st, label, str(score),
                 left_words[:], right_words[:]]]
    return res


# add missed gold spans
def fix_merged(merged):
    res = []
    for me in merged:
        found = dict(me[0][2])
        for sp in me[0][1]:
            found[sp] = found.get(sp, 0)
        res += [([me[0][0], me[0][1][:], sorted(found.items())], me[1][:])]
    return res


def crf_to_features(results_file, spans_file, labels_file, output_file):
    sentences = treat_sentences(results_file)
    print "read sentences"
    spans = treat_spans(spans_file)
    merged = fix_merged(merge(sentences, spans))
    labels = read_labels(labels_file)
    print "merged"
    f = open(output_file, 'w')
    file_name = ''
    num = 0
    for i, me in enumerate(merged):
        new_file_name = me[1][0][2]
        if new_file_name == file_name:
            num += 1
        else:
            file_name = new_file_name
            num = 0
            print >>f, ''
        rlist = treat_merged_sentence(me, labels[file_name], num)
        for ls in rlist:
            print >>f, '\t'.join(ls)
    f.close()


results_file = pjoin(git_dir, 'Detection/crfpp_output/train_res_c1_l2_v2.dat')
spans_file = pjoin(git_dir, 'Detection/crfpp_input_train/crfpp_spans_batch_1.txt')
labels_file = pjoin(git_dir, 'Detection/train_concepts1.dat')
output_file = '%s/train_concepts.dat' % (MIMIC_crf_dir,)

crf_to_features(results_file, spans_file, labels_file, output_file)

results_file = pjoin(git_dir, 'Detection/crfpp_output/dev_res_c1_l2_v2.dat')
spans_file = pjoin(git_dir, 'Detection/crfpp_input_dev/crfpp_spans_batch_1.txt')
labels_file = pjoin(git_dir, 'Detection/dev_concepts1.dat')
output_file = '%s/dev_concepts.dat' % (MIMIC_crf_dir,)

crf_to_features(results_file, spans_file, labels_file, output_file)

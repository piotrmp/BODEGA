import pathlib
import random
import sys

random.seed(1)

corpus_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

train_lines = []
test_lines = []

prev_article_id = None
article_text = ''
article_sentences = []
prop_sentences = set()
noprop_sentences = set()
for line in open(corpus_path / 'train-task1-SI.labels'):
    parts = line.strip().split('\t')
    article_id = parts[0]
    if prev_article_id != article_id:
        with open(corpus_path / 'train-articles' / ('article' + article_id + '.txt'), 'r') as file:
            article_text = file.read()
        offset = 0
        article_sentences = []
        for sentence in article_text.split('\n'):
            if len(sentence) != 0:
                article_sentences.append((offset, offset + len(sentence), sentence))
                noprop_sentences.add((article_id, offset, offset + len(sentence), sentence))
            offset = offset + len(sentence) + 1
        prev_article_id = article_id
    s_begin = int(parts[-2])
    s_end = int(parts[-1])
    span = article_text[s_begin:s_end]
    # print('--------------')
    # print(span)
    for begin, end, sentence in article_sentences:
        if not (s_begin >= end or s_end <= begin):
            sent_desc = (article_id, begin, end, sentence)
            noprop_sentences.discard(sent_desc)
            prop_sentences.add(sent_desc)
for label, sentences in [('1', prop_sentences), ('0', noprop_sentences)]:
    for sentence in sentences:
        source = 'SemEval2020T11-' + sentence[0]
        text = sentence[3]
        text = text.replace('\t', '')
        text = text.replace('\\', '\\\\')
        text = text.replace('\n', '\\n')
        digit = sentence[0][-1]
        if digit in ['4', '9']:
            subset = test_lines
        else:
            subset = train_lines
        subset.append(label + '\t' + source + '\t' + text + '\n')

div = len(test_lines) // 400
attack_lines = [line for i, line in enumerate(test_lines) if i % div == 0]
dev_lines = [line for i, line in enumerate(test_lines) if i % div != 0]

random.shuffle(train_lines)
random.shuffle(attack_lines)
random.shuffle(dev_lines)

out_train = open(out_path / 'train.tsv', 'w')
for line in train_lines:
    out_train.write(line)
out_train.close()

out_attack = open(out_path / 'attack.tsv', 'w')
for line in attack_lines:
    out_attack.write(line)
out_attack.close()

out_dev = open(out_path / 'dev.tsv', 'w')
for line in dev_lines:
    out_dev.write(line)
out_dev.close()


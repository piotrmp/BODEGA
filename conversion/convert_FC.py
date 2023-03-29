import pathlib
import random
import json
import unicodedata
import sys

random.seed(1)

corpus_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])


def normalise(string):
    return unicodedata.normalize('NFC',
                                 string.replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace(
                                     '-RSB-', ']'))


wiki_content = {}
for _filenum in range(109):
    filenum = _filenum + 1
    for line in open(corpus_path / 'wiki-pages' / 'wiki-pages' / ('wiki-' + f"{filenum:03d}" + '.jsonl')):
        struct = json.loads(line)
        if struct['id'] == '' or struct['lines'] == '':
            continue
        title = normalise(struct['id'])
        contents = {}
        for sentence in struct['lines'].split('\n'):
            parts = sentence.split('\t')
            if not parts[0].isdigit():
                continue
            sen_id = int(parts[0])
            sen_txt = normalise(parts[1])
            contents[sen_id] = sen_txt
        wiki_content[title] = contents

train_lines = []
test_lines = []
subset_matching = {'train': train_lines, 'shared_task_dev': test_lines}

for subset in subset_matching:
    for line in open(corpus_path / (subset + '.jsonl')):
        struct = json.loads(line)
        if struct['label'] == 'REFUTES':
            label = 1
        elif struct['label'] == 'SUPPORTS':
            label = 0
        else:
            continue
        claim = struct['claim']
        evidence_sets = set()
        evidence_sources = {}
        for evidence in struct['evidence']:
            evidence_set = []
            evidence_source = []
            prev_title = ''
            for x in evidence:
                title = normalise(x[2])
                if (title not in wiki_content) or (x[3] not in wiki_content[title]):
                    print("Unable to find content for: " + title + ' ' + str(x[3]))
                    break
                if title != prev_title:
                    evidence_source.append(title)
                    evidence_set.append(title.replace('_', ' ') + '.')
                evidence_set.append(wiki_content[title][x[3]])
                prev_title = title
            if len(evidence_set) > 0:
                evidence_combined = ' '.join(evidence_set)
                evidence_sets.add(evidence_combined)
                evidence_sources[evidence_combined] = '/'.join(evidence_source)
        # for evidence in evidence_sets:
        #    print(evidence)
        for evidence_set in evidence_sets:
            text = evidence_set
            text = text.replace('\t', '')
            text = text.replace('\\', '\\\\')
            text = text.replace('\n', '\\n')
            text2 = claim
            text2 = text2.replace('\t', '')
            text2 = text2.replace('\\', '\\\\')
            text2 = text2.replace('\n', '\\n')
            source = 'Wikipedia:' + evidence_sources[evidence_set]
            subset_matching[subset].append(str(label) + '\t' + source + '\t' + text + '\t' + text2 + '\n')

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

import pathlib
import random
import json
import sys

random.seed(1)

corpus_path = pathlib.Path(sys.argv[1]) #'aug-rnr-data_filtered'
out_path = pathlib.Path(sys.argv[2])

train_lines = []
test_lines = []

test1s = 0
test0s = 0
for subfolder in corpus_path.iterdir():
    event = subfolder.name
    if event.startswith('.'):
        continue
    print(event)
    if event == 'charliehebdo':
        subset = test_lines
    else:
        subset = train_lines
    for type in ['rumours', 'non-rumours']:
        subsubfolder = subfolder / type
        threads = [x.name for x in subsubfolder.iterdir() if not x.name.startswith('.')]
        threads.sort()
        for thread in threads:
            thread_text = ''
            sources = [x.name for x in (subsubfolder / thread / 'source-tweets').iterdir() if
                       not x.name.startswith('.')]
            sources.sort()
            reactions = []
            if (subsubfolder / thread / 'reactions').exists():
                reactions = [x.name for x in (subsubfolder / thread / 'reactions').iterdir() if
                             not x.name.startswith('.')]
                reactions.sort()
            t_dict = {'source-tweets': sources, 'reactions': reactions}
            first_tweet_id = None
            for origin in ['source-tweets', 'reactions']:
                for tweet in t_dict[origin]:
                    with open(subsubfolder / thread / origin / tweet) as f:
                        struct = json.load(f)
                        if 'full_text' in struct:
                            message = struct['full_text']
                        else:
                            message = struct['text']
                        message = message.replace('\n', ' ').replace('\r', ' ').replace('&amp;', '&').replace('&lt;',
                                                                                                              '<').replace(
                            '&gt;', '>')
                        thread_text = message if thread_text == '' else thread_text + '\n' + message
                        if first_tweet_id is None:
                            first_tweet_id = tweet
            text = thread_text
            text = text.replace('\t', '')
            text = text.replace('\\', '\\\\')
            text = text.replace('\n', '\\n')
            label = '1' if type == 'rumours' else '0'
            source = 'https://twitter.com/twitter/status/' + first_tweet_id
            subset.append(label + '\t' + source + '\t' + text + '\n')
            if subset is test_lines:
                if label == '1':
                    test1s += 1
                else:
                    test0s += 1

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

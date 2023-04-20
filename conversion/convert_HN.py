import pathlib
import random
import xml.sax
import sys

random.seed(1)

corpus_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

parser = xml.sax.make_parser()

EVERY_N_IN_TRAINING = 10


class LabelHandler(xml.sax.handler.ContentHandler):
    def __init__(self, every_n):
        super().__init__()
        self.labels = {}
        self.urls = {}
        self.every_n = every_n
    
    def startElement(self, name, attrs):
        if name == 'article':
            if int(attrs['id']) % self.every_n == 0:
                self.labels[attrs['id']] = 1 if (attrs['hyperpartisan'].lower() == 'true') else 0
                self.urls[attrs['id']] = attrs['url']
    
    def endElement(self, name):
        pass
    
    def characters(self, content):
        pass


class ArticleHandler(xml.sax.handler.ContentHandler):
    def __init__(self, every_n):
        super().__init__()
        self.current_article = None
        self.current_text = None
        self.articles = {}
        self.every_n = every_n
    
    def startElement(self, name, attrs):
        if name == 'article':
            if int(attrs['id']) % self.every_n == 0:
                self.current_article = attrs['id']
                self.current_text = attrs['title']+'\n'
    
    def endElement(self, name):
        if name == 'article' and self.current_article:
            self.articles[self.current_article] = self.current_text
            self.current_article = None
            self.current_text = None
    
    def characters(self, content):
        if self.current_text:
            self.current_text+=content


train_lines = []
test_lines = []
subset_dict = {'training': train_lines, 'test': test_lines}

for subset in subset_dict:
    articles_dict = {}
    date = '20181212' if subset == 'test' else '20181122'
    every_n = EVERY_N_IN_TRAINING if subset == 'training' else 1
    label_path = corpus_path / ('ground-truth-' + subset + '-bypublisher-' + date + '.xml')
    label_handler = LabelHandler(every_n)
    with open(label_path) as f:
        xml.sax.parse(f, label_handler)
    content_path = corpus_path / ('articles-' + subset + '-bypublisher-' + date + '.xml')
    content_handler = ArticleHandler(every_n)
    with open(content_path) as f:
        xml.sax.parse(f, content_handler)
    for article_id in label_handler.labels:
        label = str(label_handler.labels[article_id])
        url = label_handler.urls[article_id]
        text = content_handler.articles[article_id]
        text = text.replace('\t', '')
        text = text.replace('\\', '\\\\')
        text = text.replace('\n', '\\n')
        subset_dict[subset].append(label + '\t' + url + '\t' + text + '\n')

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

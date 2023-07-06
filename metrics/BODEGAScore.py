import OpenAttack
import numpy
from bert_score import score
import editdistance

from metrics.ScorePromise import ScorePromise
from OpenAttack.text_process.tokenizer.punct_tokenizer import PunctTokenizer

BATCH_SIZE = 16


class BODEGAScore(OpenAttack.AttackMetric):
    NAME = "BODEGA Score"
    
    def __init__(self, device, task, align_sentences=False):
        self.promises = []
        self.device = device
        self.task = task
        self.align_sentences = align_sentences
        self.tokenizerOA = PunctTokenizer()
    
    def after_attack(self, input, adversarial_sample):
        s1 = input['x']
        s2 = adversarial_sample
        promise = ScorePromise(s1, s2, self)
        self.promises.append(promise)
        return promise
    
    def normalise_for_lev(self, string):
        result = string.lower().replace('\n', ' ')
        result = self.tokenizerOA.do_detokenize([x[0] for x in self.tokenizerOA.do_tokenize(result)])
        return result
    
    def normalise_for_bert(self, string):
        if string is None:
            return 'NOT_GENERATED'
        else:
            result = string.replace('\n', ' ')
            result = self.tokenizerOA.do_detokenize([x[0] for x in self.tokenizerOA.do_tokenize(result)])
            return result
    
    def compute(self):
        alignments = [[(promise.s1, promise.s2)] for promise in self.promises]
        if self.align_sentences:
            from lambo.segmenter.lambo import Lambo
            print("Aligning sentences...")
            self.lambo = Lambo.get('English')
            alignments = self.align_sentences_greedy()
        print("Computing semantic score...")
        SS_sentences = []
        SS_guide = []
        for i, alignment in enumerate(alignments):
            SS_guide.append([])
            for sent1, sent2 in alignment:
                SS_guide[-1].append(len(SS_sentences))
                SS_sentences.append((self.normalise_for_bert(sent1), self.normalise_for_bert(sent2)))
        _, _, SS_F1_list = score([pair[0] for pair in SS_sentences], [pair[1] for pair in SS_sentences],
                                   model_type="microsoft/deberta-large-mnli",
                                   lang="en", rescale_with_baseline=True, device=self.device, batch_size=BATCH_SIZE)
        SS_F1_list = SS_F1_list.numpy()
        B_scores = []
        semantic_scores = []
        character_scores = []
        successes = []
        print("Computing BODEGA score...")
        for i, promise in enumerate(self.promises):
            if promise.s2 is None:
                # This happens if attacker gives no output or an output that doesn't succeed in changing the decision
                B_scores.append(0.0)
                successes.append(0.0)
            else:
                normalised_s1 = self.normalise_for_lev(promise.s1)
                normalised_s2 = self.normalise_for_lev(promise.s2)
                lev_dist = editdistance.eval(normalised_s1, normalised_s2)
                lev_score = 1.0 - lev_dist / max(len(normalised_s1), len(normalised_s2))
                character_scores.append(lev_score)
                semantic_score = 0
                for idx in SS_guide[i]:
                    #print(BERT_F1_list[idx])
                    #print(BERT_sentences[idx][0])
                    #print(BERT_sentences[idx][1])
                    if SS_F1_list[idx] <= 0:
                        # BERT Score is calibrated to be *usually* between 0 and 1, but not guaranteed
                        semantic_score += 0
                    elif SS_F1_list[idx] >= 1:
                        semantic_score += 1
                    else:
                        semantic_score += SS_F1_list[idx]
                semantic_score = semantic_score / len(SS_guide[i])
                semantic_scores.append(semantic_score)
                B_scores.append(semantic_score * lev_score)
                successes.append(1.0)
                # Might be useful to output most succesful attacks
                # if BERT_score * lev_score > 0.9:
                #    print(str(i + 1))
                #    print("OLD: ")
                #    print(promise.s1)
                #    print("NEW: ")
                #    print(promise.s2)
        return numpy.average(numpy.array(successes)), numpy.average(numpy.array(semantic_scores)), numpy.average(
            numpy.array(character_scores)), numpy.average(numpy.array(B_scores))
    
    def align_sentences_greedy(self):
        result = []
        for promise in self.promises:
            if promise.s2 is None:
                result.append([])
                continue
            alignment = []
            doc1_sentences = self.segment_sentences(promise.s1)
            doc2_sentences = self.segment_sentences(promise.s2)
            doc2_used_sentences = set()
            for sentence1 in doc1_sentences:
                best_sentence2 = None
                best_distance = len(promise.s1) + len(promise.s2) + 10
                for sentence2 in doc2_sentences:
                    lev_dist = editdistance.eval(self.normalise_for_lev(sentence1),
                                                 self.normalise_for_lev(sentence2)) / max(
                        len(self.normalise_for_lev(sentence1)), len(self.normalise_for_lev(sentence2)))
                    if lev_dist < best_distance:
                        best_distance = lev_dist
                        best_sentence2 = sentence2
                alignment.append((sentence1, best_sentence2))
                doc2_used_sentences.add(best_sentence2)
            # Might be useful to take into account added sentences
            # for sentence2 in doc2_sentences:
            #     if sentence2 not in doc2_used_sentences:
            #         best_sentence1 = None
            #         best_distance = len(promise.s1) + len(promise.s2) + 10
            #         for sentence1 in doc1_sentences:
            #             lev_dist = editdistance.eval(self.normalise_for_lev(sentence1),
            #                                          self.normalise_for_lev(sentence2))
            #             if lev_dist < best_distance:
            #                 best_distance = lev_dist
            #                 best_sentence1 = sentence1
            #         alignment.append((best_sentence1, sentence2))
            result.append(alignment)
        return result
    
    def segment_sentences(self, string):
        result = []
        if self.task == 'PR2':
            result = [string]
        elif self.task== 'FC':
            result = string.split('~')
        elif self.task == 'RD' or self.task =='HN':
            document = self.lambo.segment(string)
            for turn in document.turns:
                for sentence in turn.sentences:
                    result.append(sentence.text)
        return result

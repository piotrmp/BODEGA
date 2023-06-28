import OpenAttack
import numpy
from bert_score import score
import editdistance

from metrics.ScorePromise import ScorePromise

BATCH_SIZE = 16



class BODEGAScore(OpenAttack.AttackMetric):
    NAME = "BODEGA Score"
    
    def __init__(self, device, align_sentences=False):
        self.promises = []
        self.device = device
        self.align_sentences = align_sentences
    
    def after_attack(self, input, adversarial_sample):
        s1 = input['x']
        s2 = adversarial_sample
        promise = ScorePromise(s1, s2, self)
        self.promises.append(promise)
        return promise
    
    @classmethod
    def normalise_for_lev(cls, string):
        return string.lower().replace('\n', ' ').replace(' ', '')
    
    @classmethod
    def normalise_for_bert(cls, string):
        if string is None:
            return None
        else:
            return string.replace('\n', ' ')
    
    def compute(self):
        print("Computing BERT score...")
        _, _, BERT_F1 = score([str(self.normalise_for_bert(promise.s1)) for promise in self.promises],
                              [str(self.normalise_for_bert(promise.s2)) for promise in self.promises],
                              model_type="microsoft/deberta-large-mnli",
                              lang="en", rescale_with_baseline=True, device=self.device, batch_size=BATCH_SIZE)
        BERT_F1 = BERT_F1.numpy()
        B_scores = []
        BERT_scores = []
        Lev_scores = []
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
                Lev_scores.append(lev_score)
                if BERT_F1[i] <= 0:
                    # BERT Score is calibrated to be *usually* between 0 and 1, but not guaranteed
                    BERT_F1[i] = 0
                BERT_scores.append(BERT_F1[i])
                B_scores.append(BERT_F1[i] * lev_score)
                successes.append(1.0)
                if BERT_F1[i] * lev_score > 0.9:
                    print(str(i + 1))
                    print("OLD: ")
                    print(promise.s1)
                    print("NEW: ")
                    print(promise.s2)
        return numpy.average(numpy.array(successes)), numpy.average(numpy.array(BERT_scores)), numpy.average(
            numpy.array(Lev_scores)), numpy.average(numpy.array(B_scores))

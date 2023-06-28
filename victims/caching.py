import OpenAttack
import pickle
import hashlib
import numpy as np


class VictimCache(OpenAttack.Classifier):
    def __init__(self, modelPath, victim):
        self.victim = victim
        self.victim_id = str(modelPath)
        self.dictPath = modelPath.parent / (modelPath.stem + '-cache.pickle')
        if self.dictPath.exists():
            print("Victim caching: file found, loading...")
            with open(self.dictPath, 'rb') as handle:
                self.dict = pickle.load(handle)
        else:
            print("Victim caching: file not found, starting from scratch...")
            self.dict = {}
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
    
    def get_prob(self, input_):
        result = []
        for input_one in input_:
            digest = self.get_hash(input_one)
            if digest in self.dict:
                result.append(self.dict[digest])
            else:
                break
        else:
            return np.array(result)
        # Something not found, starting from scratch
        result = self.victim.get_prob(input_)
        for i, input_one in enumerate(input_):
            self.dict[self.get_hash(input_one)] = result[i]
        return result
    
    def get_hash(self, input_one):
        hash = hashlib.sha256((self.victim_id + ' # ' + input_one).encode('utf-8'), usedforsecurity=False).digest()
        return hash
    
    def finalise(self):
        print("Victim caching: saving...")
        with open(self.dictPath, 'wb') as handle:
            pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

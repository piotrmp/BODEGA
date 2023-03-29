import OpenAttack

UNK_TEXT = '🔥'

# For PWWSAttacker this is not necessary, can be fixed by token_unk=UNK_TEXT
class UnkFixWrapper(OpenAttack.Classifier):
    def __init__(self, victim):
        self.victim = victim

    def get_pred(self, input_):
        fixed_input = [x.replace('<UNK>', UNK_TEXT) for x in input_]
        return self.victim.get_pred(fixed_input)

    def get_prob(self, input_):
        fixed_input = [x.replace('<UNK>', UNK_TEXT) for x in input_]
        return self.victim.get_prob(fixed_input)

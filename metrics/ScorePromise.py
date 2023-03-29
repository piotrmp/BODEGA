class ScorePromise():
    def __init__(self, s1, s2, mainClass):
        self.s1 = s1
        self.s2 = s2
        self.mainClass = mainClass

    def __float__(self):
        return float('nan')

    def __str__(self):
        return '(later)'

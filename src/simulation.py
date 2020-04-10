TRIAL = False
VER1 = False


class SimState:
    def __init__(self):
        self.preyCount = 8000.
        self.predatorCount = 2000.

        self.environmentSustainance = 400
        self.accuracy = 0.75

        # ///////////
        # food  increase prey growth
        # water increase prey & predator growth(consumption)
        # //////////

        self.food = 10000  # 100000 kills on 30ish, 10000 kills on 100ish, 1000 is ok, 100 kills on 30ish
        self.water = 10000

        self.preyReproduction = 0.4    # Growth on best food
        self.preyDecline = 0.1  # Deaths on no food
        self.predatorDecline = 0.2  # Deaths on no food
        self.huntFactor = 0.00002
        self.predatorConsumption = 1.25

        self._accPreyDecline = 0.0
        self._accPredatorDecline = 0.0

    def step(self):
        for i in range(int(1. / self.accuracy)):

            print(self.food,  ' ', self.water)
            huntCount = 0
            futurePreyBalance = 0
            futurePredatorBalance = 0

            if self.preyCount != 0:
                huntCount = self.huntFactor * self.preyCount * self.predatorCount * 1   # P/(P+NB)
                self.preyCount -= huntCount

                if TRIAL:
                    futurePreyBalance = (0.2 - self.preyDecline) * self.preyCount
                elif VER1:
                    futurePreyBalance = ((self.food / (self.preyCount + self.food)) - self.preyDecline) * self.preyCount
                else:
                    futurePreyBalance = ((self.food / (self.preyCount + self.food) * self.preyReproduction) - self.preyDecline) * self.preyCount
            if self.predatorCount != 0:
                futurePredatorBalance = - self.predatorDecline * self.predatorCount \
                                    + huntCount * self.predatorConsumption

            self.preyCount     += futurePreyBalance     * self.accuracy
            self.predatorCount += futurePredatorBalance * self.accuracy

        self.preyCount      = max(0, int(self.preyCount))
        self.predatorCount  = max(0, int(self.predatorCount))
        return self

    def get(self):
        return self.preyCount, self.predatorCount

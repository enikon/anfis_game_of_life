from enum import Enum, auto

INF = float("inf")


class ENTITIES(Enum):
    PREY = 0
    PREDATOR = auto()


class RESOURCES(Enum):
    FOOD = 0


class SimState:
    def __init__(self):

        self.huntedDoNotFeed = False  # remove killed animals before feeding phase

        # entities: preyCount, predatorCount
        self.entityCount = [8000., 2000.]
        self.resourceLevels = [10000.0]
        self.entityCrossMatrix = [
            [(0.4, 0.1), (0.0, 0.0)],  # prey
            [(0.00002, 0.25), (0.0, 0.2)]  # predator
        ]
        # ECM[x][x] = (reproduction, decline)
        # ECM[x][y] = (hunt factor, consumption factor)
        # y is eaten by x in xy0 chance and gives xy1 food

        self.resourceCrossMatrix = [
            # food  #...
            [(1.0, 1.0)],  # prey
            [(INF, 0.0)]  # predator
        ]
        # RCM[x][y] = (significance for reproduction, competitive usage)
        # x is under condition/resource y
        # *** significance is INF = not dependant, 1 = dependant, 0.001 = needs a lot, 0 = cannot be 0

        # 100000 kills on 30ish, 10000 kills on 100ish, 1000 is ok, 100 kills on 30ish

        self.accuracy = 1.0
        self.speciesThreshold = 10

        # self.preyReproduction = 0.4    # Growth on best food
        # self.preyDecline = 0.1  # Deaths on no food
        # self.predatorDecline = 0.2  # Deaths on no food
        # self.huntFactor = 0.00002
        # self.predatorConsumption = 1.25

    def step(self):
        for i in range(int(1. / self.accuracy)):

            # zero counts
            futureBalance = [0. for j in range(len(self.entityCount))]  # what will be
            huntCount = [0. for j in range(len(self.entityCount))]  # how much it killed
            huntedCount = [0. for j in range(len(self.entityCount))]  # how much was killed
            resourcesDemand = [0. for r in range(len(self.resourceLevels))]  # how much of a resource is used up

            # update hunting & resource usage for all species
            for j in range(len(self.entityCount)):

                # if species exist
                if self.entityCount[j] > 0:

                    # all hunt for j
                    for k in range(len(self.entityCount)):
                        if j != k:
                            huntedNow = self.entityCrossMatrix[k][j][0] \
                                        * self.entityCount[j] \
                                        * self.entityCount[k]
                            huntCount[k] += huntedNow * self.entityCrossMatrix[k][j][1]
                            huntedCount[j] += huntedNow
                            # * P/(P+NB) (if grouping implemented)

                    # update consumption of resources
                    for r in range(len(self.resourceLevels)):
                        resourcesDemand[r] += (self.entityCount[j] - (huntedCount[j] if self.huntedDoNotFeed else 0)) \
                                              * self.resourceCrossMatrix[j][r][1]

            # update reproduction and consumption for all species
            for j in range(len(self.entityCount)):

                # if species exist
                if self.entityCount[j] - int(round(huntedCount[j])) > 0:
                    surviving_population = self.entityCount[j] - huntedCount[j]

                    liebigBarrel = INF
                    # find biggest restriction
                    for r in range(len(self.resourceLevels)):

                        # ignoring the not dependent resources
                        if self.resourceCrossMatrix[j][r][0] == INF:
                            liebigBarrel = min(liebigBarrel, 1.0)
                        # finding biggest restriction
                        else:
                            resourceInfluence = self.resourceLevels[r] * self.resourceCrossMatrix[j][r][0]
                            liebigBarrel = min(liebigBarrel,
                                               (resourceInfluence / (resourcesDemand[r] + resourceInfluence)))

                    # applying Lotka-Volterra
                    futureBalance[j] = \
                        (self.entityCrossMatrix[j][j][0] * liebigBarrel - self.entityCrossMatrix[j][j][1]) \
                        * surviving_population \
                        \
                        + huntCount[j]\
                        - huntedCount[j]
                else:
                    futureBalance[j] = -self.entityCount[j]

            # update living numbers
            for j in range(len(self.entityCount)):
                self.entityCount[j] += futureBalance[j] * self.accuracy

        # discretise numbers
        for j in range(len(self.entityCount)):
            self.entityCount[j] = int(round(max(0.0, self.entityCount[j])))
            if self.entityCount[j] < self.speciesThreshold:
                self.entityCount[j] = 0

        return self

    def get(self):
        return self.entityCount

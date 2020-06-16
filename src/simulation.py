from enum import Enum, auto
import math

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
            [(0.00002, 1.0), (0.0, 0.2)]  # predator
        ]
        # ECM[x][x] = (reproduction base with best conditions, absolute decline)
        # ECM[x][y] = (hunt factor (how many killed), consumption factor(how many bred from one killed)
        # y is eaten by x in xy0 chance and gives xy1 food

        self.resourceCrossMatrix = [
            # food  #...
            [(1.0, 1.0)],  # prey
            [(0.0, 0.0)]  # predator
        ]
        # RCM[x][y] = (significance for reproduction, competitive usage (how much is used))
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
        ent, res = self.step_function(self.resourceLevels)
        self.resourceLevels = res
        self.entityCount = ent
        return self

    def step_function(self, resources):

        entityOutput   = [self.entityCount[j]    for j in range(len(self.entityCount))]  # what will be
        resourceOutput = [self.resourceLevels[r] for r in range(len(self.resourceLevels))]  # how much of a resource is used up

        for i in range(int(1. / self.accuracy)):
            balance, demand = self.step_loop_function(resources)

            # update living numbers
            for j in range(len(self.entityCount)):
                entityOutput[j] = entityOutput[j] + balance[j] * self.accuracy

            # update resources numbers
            for r in range(len(self.resourceLevels)):
                resourceOutput[r] = max(0.0, resourceOutput[r] - demand[r] * self.accuracy)

        # discretise individuals numbers
        for j in range(len(self.entityCount)):
            entityOutput[j] = int(round(entityOutput[j]))
            if entityOutput[j] < self.speciesThreshold:
                entityOutput[j] = 0
        return entityOutput, resourceOutput

    def step_loop_function(self, resources):
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

            surviving_population = self.entityCount[j] - huntedCount[j]
            # if species exist
            if surviving_population > self.speciesThreshold:
                liebigBarrel = INF
                # find biggest restriction
                for r in range(len(self.resourceLevels)):

                    # ignoring the not dependent resources
                    if self.resourceCrossMatrix[j][r][0] == INF:
                        liebigBarrel = min(liebigBarrel, 1.0)
                    # finding biggest restriction
                    else:
                        resourceInfluence = resources[r] * self.resourceCrossMatrix[j][r][0]
                        liebigBarrel = min(liebigBarrel,
                                           (resourceInfluence / (resourcesDemand[r] + resourceInfluence)))

                # applying Lotka-Volterra
                futureBalance[j] = \
                    (self.entityCrossMatrix[j][j][0] * liebigBarrel - self.entityCrossMatrix[j][j][1]) \
                    * surviving_population \
                    + huntCount[j] \
                    - huntedCount[j]
            else:
                futureBalance[j] = -self.entityCount[j]

        return futureBalance, resourcesDemand

    def get(self):
        return self.entityCount.copy()

    def get_inversion(self):
        dec_prey = self.entityCrossMatrix[0][0][1]
        dec_pred = self.entityCrossMatrix[1][1][1]
        rep_prey = self.entityCrossMatrix[0][0][0]

        def func(x):
            num_prey = math.pow(10, x[0] * 6 + 1)
            num_pred = math.pow(10, x[1] * 6 + 1)

            res = -(num_prey * (dec_prey * (num_prey - dec_pred * num_pred) + dec_pred * num_pred)) / \
                   (dec_prey * (num_prey - dec_pred * num_pred) + dec_pred * rep_prey * num_pred + dec_pred * num_pred - rep_prey * num_prey)
            food = (math.log(res, 10)-1)/6
            return food

        return func

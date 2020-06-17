from enum import Enum, auto
import math

INF = float("inf")


class ENTITIES(Enum):
    PREY = 0
    PREDATOR = auto()


class RESOURCES(Enum):
    FOOD = 0


class SimState:
    def __init__(self, entities=None):

        if entities is None:
            entities = [8000., 2000.]

        self.huntedDoNotFeed = False  # remove killed animals before feeding phase

        # entities: preyCount, predatorCount
        self.entityCount = entities
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
        ent, res = self.step_function(self.entityCount, self.resourceLevels)
        self.resourceLevels = res
        self.entityCount = ent
        return self

    def step_function(self, entities, resources):

        entityOutput   = [entities[j]  for j in range(len(entities))]  # what will be
        resourceOutput = [resources[r] for r in range(len(resources))]  # how much of a resource is used up

        for i in range(int(1. / self.accuracy)):
            balance, demand = self.step_loop_function(entityOutput, resourceOutput)

            # update living numbers
            for j in range(len(entities)):
                entityOutput[j] += balance[j] * self.accuracy

            # update resources numbers
            for r in range(len(resources)):
                resourceOutput[r] = max(0., resourceOutput[r] - demand[r] * self.accuracy)

        # discretise individuals numbers
        for j in range(len(entities)):
            entityOutput[j] = int(round(entityOutput[j]))
            if entityOutput[j] < self.speciesThreshold:
                entityOutput[j] = 0

        return entityOutput, resourceOutput

    def step_loop_function(self, entities, resources):
        # zero counts
        futureBalance = [0. for j in range(len(entities))]  # what will be
        huntCount = [0. for j in range(len(entities))]  # how much it killed
        huntedCount = [0. for j in range(len(entities))]  # how much was killed
        resourcesDemand = [0. for r in range(len(resources))]  # how much of a resource is used up

        # update hunting & resource usage for all species
        for j in range(len(entities)):

            # if species exist
            if entities[j] > 0:

                # all hunt for j
                for k in range(len(entities)):
                    if j != k:
                        huntedNow = self.entityCrossMatrix[k][j][0] \
                                    * entities[j] \
                                    * entities[k]
                        huntCount[k] += huntedNow * self.entityCrossMatrix[k][j][1]
                        huntedCount[j] += huntedNow
                        # * P/(P+NB) (if grouping implemented)

                # update consumption of resources
                for r in range(len(resources)):
                    resourcesDemand[r] += (entities[j] - (huntedCount[j] if self.huntedDoNotFeed else 0)) \
                                          * self.resourceCrossMatrix[j][r][1]

        # update reproduction and consumption for all species
        for j in range(len(entities)):

            surviving_population = entities[j] - huntedCount[j]
            # if species exist
            if surviving_population > self.speciesThreshold:
                liebigBarrel = INF
                # find biggest restriction
                for r in range(len(resources)):

                    # ignoring the not dependent resources
                    if self.resourceCrossMatrix[j][r][0] == INF:
                        liebigBarrel = min(liebigBarrel, 1.0)
                    # finding biggest restriction
                    else:
                        resourceInfluence = resources[r] * self.resourceCrossMatrix[j][r][0]
                        liebigBarrel = min(liebigBarrel,
                                           (resourceInfluence / (resourcesDemand[r] + resourceInfluence))) if resourceInfluence > 0 else 0

                # applying Lotka-Volterra
                futureBalance[j] = \
                    (self.entityCrossMatrix[j][j][0] * liebigBarrel - self.entityCrossMatrix[j][j][1]) \
                    * surviving_population \
                    + huntCount[j] \
                    - huntedCount[j]
            else:
                futureBalance[j] = -entities[j]

        return futureBalance, resourcesDemand

    def get(self):
        return self.entityCount.copy()

    # TODO ADD ACCESS DECORATORS
    #HEURISTIC
    def get_inversion(self):

        def func(x):
            num_prey = int(math.pow(10, x[0] * 6 + 1))
            num_pred = int(math.pow(10, x[1] * 6 + 1))

            if num_prey < 10000:
                food = -(num_prey*(num_prey*(num_pred-45000)+500000000))/(num_prey*(num_pred - 65000)+500000000)
                if food <= 0:
                    food = num_prey*num_prey/num_pred
                    if self.step_function([num_prey, num_pred], [food])[0][0] > 10000:
                        food = (2.0 - num_prey/10000)*num_prey
            else:
                food = -10000*(num_pred+5000)/(num_pred-15000)

            if food <= 0:
                food = 0

            print("STEP FROM", [num_prey, num_pred])
            print("STEP TO", self.step_function([num_prey, num_pred], [food]))

            npy, npd = self.step_function([num_prey, num_pred], [food])

            if food <= 0:
                food = 0
            else:
                food = (math.log(food, 10)-1.0)/6
            return food

        return func

from PyNNet import PyNNet
from multiprocessing import Pool
import random
import os

#
# Trainer class to train PyNNNets by genetic algorithm constructs.
#
class PyNNTrainer(object):


    #
    # The types of evalutations available.
    # COMPETE: Each net competes against each other net once
    # SINGLE: Each net is evaluated on it's own.
    # Where possible SINGLE should be used, since it is much faster.
    # 2-player-games are main application for COMPETE.
    #
    COMPETE = True
    SINGLE  = False


    #
    # General constructor.
    # @param netsPerGen : number of nets each generation should consist of
    # @param neuronCounts : array of non-bias-neurons for each layer. First and
    #                       last layer are automatically interpreted as input
    #                       and output layers.
    # @param evaluator : an evaluator-object having at least one of the
    #                    evalutation-methods; the one specified next
    # @param competeOrSingle : determines which evalutation-method is to be
    #                          used. Boolean, but should be used with the
    #                          constants of this class.
    #                          Default : COMPETE
    # @param processes : number of processes that should be created for faster
    #                    calculation. 0 means parallel processing is
    #                    deactivated, 1 means one sub-process and thereby is not
    #                    helpful in most cases.
    #                    Default : 8
    # @param outputGeneration : Whether or not a folder for each generation
    #                           should be created, containing the data for each
    #                           entity in it.
    #                           Default : False
    #
    def __init__(self, netsPerGen, neuronCounts, evaluator,
            competeOrSingle=COMPETE, processes=8, outputGeneration=False,
            dataPath=None):
        # set output path initially
        if dataPath == None:
            self.dataPath = input("Path for data files: ")
        else:
            self.dataPath = dataPath

        # set values by params
        self.netsPerGen       = netsPerGen
        self.neuronCounts     = neuronCounts
        self.competeOrSingle  = competeOrSingle
        self.evaluator        = evaluator
        self.outputGeneration = outputGeneration
        self.processes        = processes

        # calculate the number of connections necessary to create a net
        self.numConnections = 0
        for i in range(len(self.neuronCounts) - 1):
            self.numConnections += ((self.neuronCounts[i] + 1) *
                                    (self.neuronCounts[i + 1] + 1))
                                    # "+1"s for biases
        self.numConnections -= self.neuronCounts[-1] # output layer has no bias

        # set defaults
        self.currentGen  = 0
        self.nets        = None
        self.fitnessList = [None for _ in range(netsPerGen)] # create empty

        # set parameters to defaults:
        self.set_chance_mutation_mutateConnection()
        self.set_chance_mutation_totalChange()
        self.set_chance_generateByParents_copyParent()
        self.set_generateGeneration_copyBests()
        self.set_generateGeneration_copyBestWithoutMutation()
        self.set_generateGeneration_removeWorsts()


    #
    # Evaluates the fitnesses of the current nets.
    # Uses the parameters specified beforehand.
    #
    def evalCurrentGen(self):
        if (self.processes > 0): # if parallel
            # create list of parameters to map with multiprocessing-Pool
            params = []

            # if COMPETE, add those params
            if self.competeOrSingle == PyNNTrainer.COMPETE:
                # loop through every combination of nets (each combination once)
                for net1 in range(self.netsPerGen):
                    for net2 in range(net1 + 1, self.netsPerGen):
                        params.append([
                            self.getNetFromIndex(net1),
                            self.getNetFromIndex(net2),
                            net1,
                            net2
                        ])
            # if SINGLE, add those params
            elif self.competeOrSingle == PyNNTrainer.SINGLE:
                for netIndex in range(self.netsPerGen):
                    params.append(self.getNetFromIndex(netIndex))

            # create pool of sub-processes
            pool = Pool(processes=self.processes)

            # if COMPETE, run that function with the pool and interpret results
            if self.competeOrSingle == PyNNTrainer.COMPETE:
                data = pool.map(self.evaluator.competeNets, params)
                pool.close()

                # reset fitnessList
                self.fitnessList = [0 for _ in range(self.netsPerGen)]

                for res in data:
                    # [(net1Index, net1Fitness), (net2Index, net2Fitness)]
                    self.fitnessList[res[0][0]] += res[0][1]
                    self.fitnessList[res[1][0]] += res[1][1]
            # if SINGLE, run that function
            elif self.competeOrSingle == PyNNTrainer.SINGLE:
                self.fitnessList = pool.map(self.evaluator.testFitness, params)
                pool.close()
        else: # if not parallel
            # if COMPETE, run that function and interpret results
            if self.competeOrSingle == PyNNTrainer.COMPETE:
                data = []

                # loop through every combination of nets (each combination once)
                for net1 in range(self.netsPerGen):
                    for net2 in range(net1 + 1, self.netsPerGen):
                        data.append(self.evaluator.competeNets([
                            self.getNetFromIndex(net1),
                            self.getNetFromIndex(net2),
                            net1,
                            net2
                        ]))

                # reset fitnessList
                self.fitnessList = [0 for _ in range(self.netsPerGen)]

                for res in data:
                    # [(net1Index, net1Fitness), (net2Index, net2Fitness)]
                    self.fitnessList[res[0][0]] += res[0][1]
                    self.fitnessList[res[1][0]] += res[1][1]
            # if SINGLE, run that function
            elif self.competeOrSingle == PyNNTrainer.SINGLE:
                for netIndex in range(self.netsPerGen):
                    self.fitnessList[netIndex] = self.evaluator.testFitness(
                        self.getNetFromIndex(netIndex)
                    )

        # if outputGeneration is activated, write fitnesses
        """
        (create/write)
        median-fitness
        index fitness
        index fitness...
        """
        if self.outputGeneration:
            filename = (self.dataPath + "/pynntrainer_" +
                        str(self.getCurrentGenNum()) + "/pynntrainer_evaldata")
            os.makedirs(os.path.dirname(filename), exist_ok = True)
            with open(filename, "w") as f:
                f.write(str(sum(self.fitnessList) / self.netsPerGen) + "\n")
                for netIndex in range(self.netsPerGen):
                    f.write(str(netIndex) + " " +
                        str(self.fitnessList[netIndex]) + "\n")

        # append this generation's fitness data to file for all generations.
        """
        (append)
        genNum max med min
        """
        filename = self.dataPath + "/pynntrainer_evaldata"
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        with open(filename, "a") as f:
            f.write(str(self.getCurrentGenNum()) + " " +
                    str(max(self.fitnessList)) + " " +
                    str(sum(self.fitnessList) / self.netsPerGen) + " " +
                    str(min(self.fitnessList)) + "\n")


    #
    # Generate the next generation, based on the last (if there was one).
    # Uses the externally and previously set parameters.
    #
    def genNextGen(self):
        if self.getCurrentGenNum() == 0:
            # generate first gen randomly
            self.nets = [[random.random() for _ in range(self.numConnections)]
                            for __ in range(self.netsPerGen)]
            # # SAME AS (optimized)
            # self.nets = []
            # for i in range(self.netsPerGen):
            #     net = []
            #     for c in range(numConnections):
            #         net.append(random.random()) # weight of connection [0;1[
            #     self.nets.append(net)
        else:
            # generate next gen
            self.__h_generateNextGen()

            # mutate some nets to get better/different results
            self.__h_mutate()

        self.currentGen += 1

        # if outputGeneration: write each net to file
        if self.outputGeneration:
            for i, net in enumerate(self.nets):
                filename = (self.dataPath + "/pynntrainer_" +
                            str(self.getCurrentGenNum()) +
                            "/pynntrainer_" + str(i))
                os.makedirs(os.path.dirname(filename), exist_ok = True)
                with open(filename, "w") as f:
                    f.write(" ".join(map(str, net)))


    #
    # Generate next generation based on last.
    # Uses previously specified parameters.
    #
    def __h_generateNextGen(self):
        oldNets = list(self.nets)
        self.nets = []

        # copy best into nets (protection from mutation is in mutation func)
        if self.get_generateGeneration_copyBestWithoutMutation():
            self.nets.append(oldNets[
                self.fitnessList.index(max(self.fitnessList))])
            # # SAME AS (optimized)
            # indexMax = self.fitnessList.index(max(self.fitnessList))
            # net = oldNets[indexMax]
            # self.nets.append(net)

        # copy bests (with mutation)
        if self.get_generateGeneration_copyBests()[0] > 0:
            bestIndizes = []
            bestFitnesses = []
            for index, fitness in enumerate(self.fitnessList):
                if (len(bestIndizes) <
                    self.get_generateGeneration_copyBests()[0]):
                    bestIndizes.append(index)
                    bestFitnesses.append(fitness)
                    break # go to next, skip rest of loop
                minimum = min(bestFitnesses)
                if minimum < fitness:
                    selIndex = bestFitnesses.index(minimum)
                    bestFitnesses[selIndex] = fitness
                    bestIndizes[selIndex] = index
            # add the nets to next generation
            for i in bestIndizes:
                # add as often as specified (to protect from mutation)
                for m in range(self.get_generateGeneration_copyBests()[1]):
                    self.nets.append(oldNets[i])

        # all possible parent-indizes, with their respective weight (fitness)
        indexChoices = list(enumerate(self.fitnessList))

        # remove n worst from possibilities
        for _ in range(self.get_generateGeneration_removeWorsts()):
            indexChoices.remove(min(indexChoices))

        if self.processes > 0: # if parallel
            params = [
            (oldNets, indexChoices) for _ in
                range(self.netsPerGen - len(self.nets)) # missing amount
            ]
            pool = Pool(processes=self.processes)
            newNets = pool.map(self.h_generateNewNetsByParents, params)
            pool.close()
            self.nets += newNets
        else: # if not parallel
            while len(self.nets) < self.netsPerGen:
                self.nets.append(self.__h_generateNetPyParents(
                    oldNets[PyNNTrainer.__h_weightedChoice(indexChoices)],
                    oldNets[PyNNTrainer.__h_weightedChoice(indexChoices)]
                ))
                # # SAME AS (optimized)
                # i1  = PyNNTrainer.__h_weightedChoice(indexChoices)
                # i2  = PyNNTrainer.__h_weightedChoice(indexChoices)
                # net = self.__h_generateNetPyParents(oldNets[i1], oldNets[i2])
                # self.nets.append(net)

    #
    # HELPER FUNCTION - SHOULD BE PRIVATE, CAN'T BECAUSE OF PARALLELISM!!!
    #
    # Creates a new net based on two weighted-random parents.
    #
    def h_generateNewNetsByParents(self, params):
        return self.__h_generateNetPyParents(
            params[0][PyNNTrainer.__h_weightedChoice(params[1])],
            params[0][PyNNTrainer.__h_weightedChoice(params[1])]
        )
        # # SAME AS (optimized)
        # oldNets = params[0]
        # indexChoices = params[1]
        # return self.__h_generateNetPyParents(
        #     oldNets[PyNNTrainer.__h_weightedChoice(indexChoices)],
        #     oldNets[PyNNTrainer.__h_weightedChoice(indexChoices)]
        # )


    #
    # Generates a new net based off two parent-nets.
    # Uses previously defined parameters.
    #
    def __h_generateNetPyParents(self, parent1, parent2):
        # handle copying-case
        if random.random() <= self.get_chance_generateByParents_copyParent():
            if random.random() < .5: # 50:50 for each parent
                return parent1
            return parent2

        r = random.randint(1, self.numConnections - 1)
        return parent1[:r] + parent2[r:]


    #
    # Mutates the current nets, using previously specified parameters.
    #
    def __h_mutate(self):
        if self.processes > 0: # if parallel
            pool = Pool(processes=self.processes)

            # data = pool.map(function, mapArray)
            pool.map(self.h_mutateNet, (range(1, len(self.nets))
                if self.get_generateGeneration_copyBestWithoutMutation()
                else range(len(self.nets))))
            # # SAME AS (optimized)
            # params = (range(1, len(self.nets))
            #             if self.get_generateGeneration_copyBestWithoutMutation()
            #             else range(len(self.nets)))
            # pool.map(self.h_mutateNet, params)

            pool.close()
        else: # if not parallel
            # loop through each connection in each net
            for netIndex in (range(1, len(self.nets))
                if self.get_generateGeneration_copyBestWithoutMutation()
                else range(len(self.nets))):
            # # SAME AS (optimized)
            # # get range previously, to protect first item (best net from before)
            # # from mutation, if wanted
            # netRange = ( range(1, len(self.nets))
            #     if self.get_generateGeneration_copyBestWithoutMutation()
            #     else range(len(self.nets)) )
            #
            # for netIndex in netRange:
                for connectionIndex in range(self.numConnections):
                    # if chosen for mutation, mutate
                    if (random.random() <=
                        self.get_chance_mutation_mutateConnection()):
                        # if chosen to change completely randomly, do so
                        if random.random() > self.get_chance_mutation_totalChange():
                            self.nets[netIndex][connectionIndex] = random.random()
                        # mutate based on previous value, if chosen for this
                        else:
                            r = random.uniform(-1, 1)
                            prev = self.nets[netIndex][connectionIndex]
                            factor = 1# min([r * (1 - prev), r * prev])
                            r *= factor
                            self.nets[netIndex][connectionIndex] += r


    #
    # HELPER FUNCTION - SHOULD BE PRIVATE, CAN'T BECAUSE OF PARALLELISM!!!
    #
    # Loops through all connections of the specified net and mutates then based
    # on previously set parameters.
    #
    def h_mutateNet(self, netIndex):
        # loop through all connections of this net
        for connectionIndex in range(self.numConnections):
            # if chosen for mutation, mutate
            if (random.random() <=
                self.get_chance_mutation_mutateConnection()):
                # if chosen to change completely randomly, do so
                if random.random() > self.get_chance_mutation_totalChange():
                    self.nets[netIndex][connectionIndex] = random.random()
                # mutate based on previous value, if chosen for this
                else:
                    r = random.uniform(-1, 1)
                    prev = self.nets[netIndex][connectionIndex]
                    factor = 1# min([r * (1 - prev), r * prev])
                    r *= factor
                    self.nets[netIndex][connectionIndex] += r


    #
    # Return a PyNNet-object based on the connections in nets-list and
    # parameters set previously.
    #
    def getNetFromIndex(self, netIndex):
        return PyNNet(self.neuronCounts, self.nets[netIndex], tanh=True)


    #
    # The current generation's number. (First generation is 1)
    #
    def getCurrentGenNum(self):
        return self.currentGen


    #
    # Start from where was left off.
    # This also gives the possibility to go back a few generations if the nets
    # are getting worse, changing a few parameters and then seeing how that
    # changes the outcome.
    # @param generationNumber : The number of the generation to read in and
    #                           continue developing with. ()
    #
    def loadStateFromFiles(self, generationNumber):
        print("TODO: <PyNNTrainer-obj>.loadStateFromFiles(generationNumber)")


    #
    # --------------------------------------------------------------------------
    # Getters and setters for all params that influence the run-through.
    # They will take action as fast as possible (though that my sometimes be
    # one generation ahead).
    #

    def set_chance_mutation_mutateConnection(self, chance=.002):
        self.__param_chance_mutation_mutateConnection = chance

    def get_chance_mutation_mutateConnection(self):
        return self.__param_chance_mutation_mutateConnection

    def set_chance_mutation_totalChange(self, chance=.02):
        self.__param_chance_mutation_totalChange = chance

    def get_chance_mutation_totalChange(self):
        return self.__param_chance_mutation_totalChange

    def set_chance_generateByParents_copyParent(self, chance=.3):
        self.__param_chance_generateByParents_copyParent = chance

    def get_chance_generateByParents_copyParent(self):
        return self.__param_chance_generateByParents_copyParent

    def set_generateGeneration_copyBests(self, bests=0, times=1):
        self.__param_generateGeneration_copyBests = (bests, times)

    def get_generateGeneration_copyBests(self):
        return self.__param_generateGeneration_copyBests

    def set_generateGeneration_copyBestWithoutMutation(self, copy=False):
        self.__param_generateGeneration_copyBestWithoutMutation = copy

    def get_generateGeneration_copyBestWithoutMutation(self):
        return self.__param_generateGeneration_copyBestWithoutMutation

    def set_generateGeneration_removeWorsts(self, worsts=0):
        self.__param_generateGeneration_removeWorsts = worsts

    def get_generateGeneration_removeWorsts(self):
        return self.__param_generateGeneration_removeWorsts

    #
    # s.O.
    # --------------------------------------------------------------------------
    #


    #
    # Chooses one of the choices (which are the keys/indizes) and returns that,
    # based on the weight it was given for the choice (which is the value).
    # Meaning that if you give the following array:
    # [ 5, 3, 1, 1 ]
    # 0 has a 50% change of being chosen, as opposed to 1 (30%), 2 (10%) and
    # 3 (10%).
    #
    def __h_weightedChoice(choices):
        r = random.uniform(0, sum(w for c, w in choices))
        # SAME AS (optimized)
        # total = sum(w for c, w in choices) # sum of weights
        # r = random.uniform(0, total)

        for choice, weight in choices:
            r -= weight
            if 0 >= r:
                return choice
        assert False, "Shouldn't get here"
        # # ORIGINAL
        # total = sum(w for c, w in choices)
        # r = random.uniform(0, total)
        # upto = 0
        # for c, w in choices:
        #     if upto + w >= r:
        #         return c
        #     upto += w
        # assert False, "Shouldn't get here"


#
# Parent-class for evalutators. Contains specifications as to how the different
# evalution-functions should work.
#
class PyNNEvaluator(object):

    # params = [net1, net2, net1Index, net2Index]
    def competeNets(self, params):
        pass # return [(net1Index, net1Fitness), (net2Index, net2Fitness)]

    def testFitness(self, net):
        pass # return netFitness

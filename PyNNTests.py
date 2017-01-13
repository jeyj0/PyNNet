import PyNNet
from PyNNTrainer import PyNNTrainer, PyNNEvaluator
import random

# TEST
import colorsys


def getFitness(r, g, b, net):
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h = int(h * 255)
    s = int(s * 255)
    v = int(v * 255)

    res = net.calc(getNumAsBinary(r, 8) +
                    getNumAsBinary(g, 8) +
                    getNumAsBinary(b, 8))
    for i in range(len(res)):
        res[i] = 1 if res[i] > .5 else 0

    rh = getBinAsNum(res[:8])
    rs = getBinAsNum(res[8:16])
    rv = getBinAsNum(res[16:])

    difference = abs(rh - h) + abs(rs - s) + abs(rv - v)
    # max-difference: 255 * 3 => 100%
    # the closer, the better => 0 difference is best

    fitness = (100 - (difference / 7.65)) ** 2 / 100
    # # SAME AS (optimized)
    # fitness = difference / (255 * 3 / 100)
    # fitness = 100 - fitness
    # fitness = fitness**2 / 100 # make logarithmic

    # to be safe put fitness in range [0;100]
    fitness = fitness if fitness <= 100 else 100
    fitness = fitness if fitness >=   0 else   0
    return fitness


def getNumAsBinary(num, length=8):
    l = [int(x) for x in bin(num)[2:]]
    if len(l) > length:
        raise ValueError("Length " + str(length) + " too short for value " + str(num) + ".")
    while len(l) < length:
        l.insert(0, 0)
    return l


def getBinAsNum(l):
    l.reverse()
    n = 0
    for i, v in enumerate(l):
        n += v * 2**i
    return n


class TestEvaluator(PyNNEvaluator):
    def testFitness(self, net):
        rs = [0, .25, .5, .75, 1]
        gs = [0, .25, .5, .75, 1]
        bs = [0, .25, .5, .75, 1]

        # rs = [0, .5, 1]
        # gs = [0, .5, 1]
        # bs = [0, .5, 1]

        fitnesses = []

        for r in rs:
            for g in gs:
                for b in bs:
                    fitnesses.append(getFitness(r, g, b, net))

        return sum(fitnesses) / len(fitnesses)

evaluator = TestEvaluator()

test_gens = 2000

trainer = PyNNTrainer(160, [24, 72, 48, 24], evaluator, PyNNTrainer.SINGLE,
                        processes=8)

# trainer.set_chance_mutation_mutateConnection(.004)
# trainer.set_chance_mutation_totalChange(.04)
# trainer.set_chance_generateByParents_copyParent(.6)
# trainer.set_generateGeneration_copyBests(5, 2)
trainer.set_generateGeneration_copyBestWithoutMutation(copy=True)
trainer.set_generateGeneration_removeWorsts(5)

trainer.outputGeneration = True

import time

fullstart = time.time()
average = 0
for i in range(test_gens):

    if i == 0:
        print("No need to cool down now ;)")
    elif i % 100 == 0:
        print("Taking a 5min break from processing, to cool down again :)")
        time.sleep(5*60)
    elif i % 50 == 0:
        print("Taking a 2.5min break from processing, to cool down again :)")
        time.sleep(2.5*60)
    elif i % 10 == 0:
        print("Taking a 30s break from processing, to cool down again :)")

    print(i+1, "...")

    # output last generation (should contain best net overall)
    # if i == test_gens - 1:
    #     trainer.outputGeneration = True

    start = time.time()
    trainer.genNextGen()
    trainer.evalCurrentGen()
    end = time.time()

    print("needed:", end - start)
    average = (average * i + (end-start)) / (i+1) if average > 0 else end - start
    left = average * (test_gens - (i+1))
    print("left (approx.) sec:", left)
    print("left (approx.) min:", left / 60)
    print("left (approx.): " + str(int(left // 3600)) + ":" +
                                str(int((left % 3600) // 60)) + ":" +
                                str(int(((left % 3600) % 60) % 60)))

    barlength = 100
    l = "["
    filled = int((i / test_gens) * barlength)
    for _ in range(filled):
        l += "#"
    for _ in range(barlength - filled):
        l += " "
    l += "]"
    print(l)
print("Done.")
completedIn = time.time() - fullstart
print("Needed:", str(int(completedIn // 3600)) + ":" +
                    str(int((completedIn % 3600) // 60)) + ":" +
                    str(int(((completedIn % 3600) % 60) % 60)))

print("Best net in the end")

print(trainer.nets[trainer.fitnessList.index(max(trainer.fitnessList))])

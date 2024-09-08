import math
import monkdata as m
import random
import matplotlib.pyplot as plt
import numpy as np
# import standard deviation function
from statistics import stdev

def entropy(dataset):
    "Calculate the entropy of a dataset"
    n = len(dataset)
    nPos = len([x for x in dataset if x.positive])
    nNeg = n - nPos
    if nPos == 0 or nNeg == 0:
        return 0.0
    return -float(nPos)/n * log2(float(nPos)/n) + \
        -float(nNeg)/n * log2(float(nNeg)/n)


def averageGain(dataset, attribute):
    "Calculate the expected information gain when an attribute becomes known"
    weighted = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        weighted += entropy(subset) * len(subset)
    return entropy(dataset) - weighted/len(dataset)


def log2(x):
    "Logarithm, base 2"
    return math.log(x, 2)


def select(dataset, attribute, value):
    "Return subset of data samples where the attribute has the given value"
    return [x for x in dataset if x.attribute[attribute] == value]


def bestAttribute(dataset, attributes):
    "Attribute with highest expected information gain"
    gains = [(averageGain(dataset, a), a) for a in attributes]
    return max(gains, key=lambda x: x[0])[1]


def allPositive(dataset):
    "Check if all samples are positive"
    return all([x.positive for x in dataset])


def allNegative(dataset):
    "Check if all samples are negative"
    return not any([x.positive for x in dataset])


def mostCommon(dataset):
    "Majority class of the dataset"
    pCount = len([x for x in dataset if x.positive])
    nCount = len([x for x in dataset if not x.positive])
    return pCount > nCount


class TreeNode:
    "Decision tree representation"

    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        "Produce readable (string) representation of the tree"
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'


class TreeLeaf:
    "Decision tree representation for leaf nodes"

    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        "Produce readable (string) representation of this leaf"
        if self.cvalue:
            return '+'
        return '-'


def buildTree(dataset, attributes, maxdepth=1000000):
    "Recursively build a decision tree"

    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        if allPositive(dataset):
            return TreeLeaf(True)
        if allNegative(dataset):
            return TreeLeaf(False)
        return buildTree(dataset, attributes, maxdepth-1)

    default = mostCommon(dataset)
    if maxdepth < 1:
        return TreeLeaf(default)
    a = bestAttribute(dataset, attributes)
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft))
                for v in a.values]
    return TreeNode(a, dict(branches), default)


def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[sample.attribute[tree.attribute]], sample)


def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)


def allPruned(tree):
    "Return a list of trees, each with one node replaced by the corresponding default class"
    if isinstance(tree, TreeLeaf):
        return ()
    alternatives = (TreeLeaf(tree.default),)
    for v in tree.branches:
        for r in allPruned(tree.branches[v]):
            b = tree.branches.copy()
            b[v] = r
            alternatives += (TreeNode(tree.attribute, b, tree.default),)
    return alternatives


def partition(data, fraction):
    "Partition data into two sets of given fraction"
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def main():
    
    # -------- ASSIGNMENT 1 ---------
    # print("Entropy of the training set monk1: %.5f" % entropy(m.monk1))
    # print("Entropy of the training set monk2: %.5f" % entropy(m.monk2))
    # print("Entropy of the training set monk3: %.5f" % entropy(m.monk3))

    # -------- ASSIGNMENT 3 ---------
    # for i in range(6):
    #     print("Information gain in attribute %d training set monk1: %.5f" % (i, averageGain(m.monk1, m.attributes[i])))
   
    # print("\n")
    # for i in range(6):
    #     print("Information gain in attribute %d training set monk2: %.5f" % (i, averageGain(m.monk2, m.attributes[i])))
    # print("\n")
    # for i in range(6):
    #     print("Information gain in attribute %d training set monk3: %.5f" % (i, averageGain(m.monk3, m.attributes[i])))

    # -------- ASSIGNMENT 5 ---------
    # split monk1 dataset
    best_attribute_monk1 = bestAttribute(m.monk1, m.attributes)
    print("Best attribute in monk1 dataset: %s" % best_attribute_monk1)
    subset1_monk1 = select(m.monk1, m.attributes[4], 1)
    subset2_monk1 = select(m.monk1, m.attributes[4], 2)
    subset3_monk1 = select(m.monk1, m.attributes[4], 3)
    subset4_monk1 = select(m.monk1, m.attributes[4], 4)

    # information gain on all attributes in subset1_monk1
    for i in range(1, 5):
        for j in range(6):
            if j == 4:
                continue
            print("Information gain in attribute %d training set where a5 == %d: %.5f" % (j+1, i,averageGain(select(m.monk1, m.attributes[4], i), m.attributes[j])))
        print("\n")

    # highest information gain in subset2_monk1 == a4
    sub1subset2_monk1 = select(subset2_monk1, m.attributes[3], 1)
    sub2subset2_monk1 = select(subset2_monk1, m.attributes[3], 2)
    sub3subset2_monk1 = select(subset2_monk1, m.attributes[3], 3)
    
    # highest information gain in subset3_monk1 == a6
    sub1subset3_monk1 = select(subset3_monk1, m.attributes[5], 1)
    sub2subset3_monk1 = select(subset3_monk1, m.attributes[5], 2)

    # highest information gain in subset4_monk1 == a1
    sub1subset4_monk1 = select(subset4_monk1, m.attributes[0], 1)
    sub2subset4_monk1 = select(subset4_monk1, m.attributes[0], 2)
    sub3subset4_monk1 = select(subset4_monk1, m.attributes[0], 3)

    # assign majority class to each subset
    print("Majority class in subset1_monk1: %s" % mostCommon(subset1_monk1))
    print("Majority class in subset2_monk1: %s" % mostCommon(sub1subset2_monk1))
    print("Majority class in subset3_monk1: %s" % mostCommon(sub2subset2_monk1))
    print("Majority class in subset4_monk1: %s" % mostCommon(sub3subset2_monk1))
    print("Majority class in subset5_monk1: %s" % mostCommon(sub1subset3_monk1))
    print("Majority class in subset6_monk1: %s" % mostCommon(sub2subset3_monk1))
    print("Majority class in subset7_monk1: %s" % mostCommon(sub1subset4_monk1))
    print("Majority class in subset8_monk1: %s" % mostCommon(sub2subset4_monk1))
    print("Majority class in subset9_monk1: %s" % mostCommon(sub3subset4_monk1))    
    
    # build tree using our own subsets and majority class
    t1 = TreeNode(m.attributes[4], {1: TreeLeaf(True), 
                                    2: TreeNode(m.attributes[3], {1: TreeLeaf(False), 2: TreeLeaf(False), 3: TreeLeaf(False)}, False), 
                                    3: TreeNode(m.attributes[5], {1: TreeLeaf(False), 2: TreeLeaf(False)}, False), 
                                    4: TreeNode(m.attributes[0], {1: TreeLeaf(False), 2: TreeLeaf(False), 3: TreeLeaf(True)}, 
                                    False)}, False)
    
    # check t1
    print(check(t1, m.monk1test))
    # Compare with built tree using buildTree function
    t_monk1 = buildTree(m.monk1, m.attributes)
    print("Monk 1 test: %0.5f" % check(t_monk1, m.monk1test))
    print("Monk 1 train: %0.5f" % check(t_monk1, m.monk1))
    # also do this for monk2 and monk3 datasets
    t_monk2 = buildTree(m.monk2, m.attributes)
    print("Monk 2 test: %0.5f" % check(t_monk2, m.monk2test))
    print("Monk 2 train: %0.5f" % check(t_monk2, m.monk2))

    t_monk3 = buildTree(m.monk3, m.attributes)
    print("Monk 3 test: %0.5f" % check(t_monk3, m.monk3test))
    print("Monk 3 train: %0.5f" % check(t_monk3, m.monk3))
    # Information gains for monk1 dataset to results

    # -------- ASSIGNMENT 6 ---------
    # Pruning

    # break when all the prunded trees are worse than the current candidate in accuracy with monk1val
    monk1train, monk1val = partition(m.monk1, 0.6)
    t = buildTree(monk1train, m.attributes)

    print("Unpruned tree monk1: %0.5f" % check(t, monk1val))
    while True:
        candidates = allPruned(t)
        for alternative in candidates:
            if check(alternative, monk1val) > check(t, monk1val):
                t = alternative
        # if all pruned trees are worse than the current candidate, break
        if all(check(a, monk1val) <= check(t, monk1val) for a in candidates):
            break
    
    print("Pruned tree monk1: %0.5f" % check(t, m.monk1test))

    # -------- ASSIGNMENT 7 ---------
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    accuricies_monk1 = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: []}
    accuricies_monk3 = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: []}
    for i in range(500):

        for f in fractions:
            monk1train, monk1val = partition(m.monk1, f) # random shuffle
            monk3train, monk3val = partition(m.monk3, f) # random shuffle
            t_1 = buildTree(monk1train, m.attributes)
            t_3 = buildTree(monk3train, m.attributes)

            # monk1
            while True:
                candidates_monk1 = allPruned(t_1)
                for alternative_monk1 in candidates_monk1:
                    if check(alternative_monk1, monk1val) > check(t_1, monk1val):
                        t_1 = alternative_monk1
                if all(check(a, monk1val) <= check(t_1, monk1val) for a in candidates_monk1):
                    break
            accuricies_monk1[f].append(check(t_1, m.monk1test))

            # monk3
            while True:
                candidates_monk3 = allPruned(t_3)
                for alternative_monk3 in candidates_monk3:
                    if check(alternative_monk3, monk3val) > check(t_3, monk3val):
                        t_3 = alternative_monk3
                if all(check(a, monk3val) <= check(t_3, monk3val) for a in candidates_monk3):
                    break
            accuricies_monk3[f].append(check(t_3, m.monk3test))

    statistics_monk1 = {}
    statistics_monk3 = {}
    for f in fractions:
        #monk1
        mean_monk1 = 1 - sum(accuricies_monk1[f])/len(accuricies_monk1[f])
        std_monk1 = stdev(accuricies_monk1[f])
        statistics_monk1[f] = (mean_monk1, std_monk1)
        #monk3
        mean_monk3 = 1 - sum(accuricies_monk3[f])/len(accuricies_monk3[f])
        std_monk3 = stdev(accuricies_monk3[f])
        statistics_monk3[f] = (mean_monk3, std_monk3)

    for f in fractions:
        print("Monk 1 - Fraction: %0.1f Mean error: %0.5f Std: %0.5f" % (f, statistics_monk1[f][0], statistics_monk1[f][1]))
    print("\n")
    for f in fractions:
        print("Monk 3 - Fraction: %0.1f Mean error: %0.5f Std: %0.5f" % (f, statistics_monk3[f][0], statistics_monk3[f][1]))
    
    # plot the mean error in a linegraph for each fraction for monk1 and monk3
    plt.figure()
    plt.plot(fractions, [statistics_monk1[f][0] for f in fractions], label='Monk1 mean error', marker='o')
    plt.plot(fractions, [statistics_monk3[f][0] for f in fractions], label='Monk3 mean error', marker='o')
    plt.xlabel('Fraction')
    plt.ylabel('Mean error')
    plt.title('Mean error for different fractions in monk1 and monk3')
    plt.legend()
    plt.grid(True)  # Add grid lines
    plt.show()

    # plot the standard deviation in a linegraph for each fraction for monk1 and monk3
    plt.figure()
    plt.plot(fractions, [statistics_monk1[f][1] for f in fractions], label='Monk1 std', marker='o')
    plt.plot(fractions, [statistics_monk3[f][1] for f in fractions], label='Monk3 std', marker='o')
    plt.xlabel('Fraction')
    plt.ylabel('Standard deviation')
    plt.title('Standard deviation for different fractions in monk1 and monk3')
    plt.legend()
    plt.grid(True)  # Add grid lines
    plt.show()

if __name__ == "__main__":        
    main()
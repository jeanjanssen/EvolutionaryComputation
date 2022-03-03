import numpy as np
import random
import math
from matplotlib import pyplot as plt

#TODO write tournament selection, and roulette selection
#TODO write 2 point crossover
#TODO diversity fitness stuff


#EXPERIMENTS
#TODO popsize 25-200, with increases of 25
#TODO probability of crossover and mutation, 0-1, with increases of 0.2
#TODO elitism proportion, 0-10, increases of 2
#TODO number of generations,10k-50k with increases of 10k



# index = np.arange(1, 10)
# weight = np.random.randint(1, 10)
# set on half average value of ind_size
max_weight = 75
# value = index
ind_size=30
pop_size=30
num_generations = 10000

p_cross = 0.4
p_mut = 0.4
def matrixgen(size):
    matrix = []
    for i in range(1,size+1):
        matrix.append([i, random.randint(1,10)])
    return matrix


class Population:
    def __init__(self, popind):
        self.pop = np.random.randint(2, size=popind)
    def fitness_calc(self, matrix):
        self.fitness = []
        for i in range(pop_size):
            solmat = np.dot(self.pop[i],matrix)
            if solmat[1]<= max_weight:
                self.fitness.append(solmat[0])
            else:
                self.fitness.append(0)

    #Form of elitist selection
    def elitist_selection(self, top=2):

        n=len(self.fitness)
        temp = np.array(self.fitness)
        maxindices = (-temp).argsort()[:n]

        newPop = Population((pop_size-top, ind_size))
        best = maxindices[:top]
        i=0
        while len(newPop.pop)<pop_size:

            # crossover
            ind = self.pop[maxindices[i]]
            if i != 0:
                if random.uniform(0, 1) < p_cross:
                    # a = random.randint(0, top-1)
                    a = random.randint(0, len(self.pop)-1)

                    ind = self.single_point_crossover(ind, self.pop[maxindices[a]])
                    #ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

                #mutation
                if random.uniform(0,1) < p_mut:
                    ind = self.single_bit_mutation(ind)

            #add to pop
            newPop.pop = np.append(newPop.pop, [ind], axis=0)
        self.pop = newPop.pop
    def single_point_crossover(self, ind1, ind2):
        point = random.randint(0, len(ind1))
        child = ind1.copy()
        for i in range(point,len(ind1)):
            child[i] = ind2[i]
        return child
    # def two_point_crossover(self, ind1, ind2):

    def single_bit_mutation(self, ind):
        a = random.randint(0, len(ind)-1)
        if ind[a] == 0:
            ind[a] = 1
        else:
            ind[a] = 0
        return ind

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    popind = (pop_size, ind_size)
    test = Population(popind)
    a = matrixgen(ind_size)

    test.fitness_calc(a)
    print(test.pop)
    print(test.fitness)
    i = 0
    plotlist = []
    plotlist1 = []
    while  i < num_generations:
        test.elitist_selection(top=10)
        test.fitness_calc(a)
        plotlist.append(max(test.fitness))
        plotlist1.append(sum(test.fitness)/len(test.fitness))
        i = i + 1
    print(test.pop)
    print(test.fitness)
    plt.plot(plotlist)
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.plot(plotlist1)
    plt.show()


import numpy as np
import random
import math

#TODO Selection method
#TODO Mutation method
#TODO Jean for TSP

#TODO experimenting with different stuff
#TODO Tournament selection, elitist and Roulette
#TODO Single point, 2 point
#TODO single bit mutation
#TODO probabilities crossover, mutation
#TODO number generations, population size


# index = np.arange(1, 10)
# weight = np.random.randint(1, 10)
max_weight = 25
# value = index
ind_size=10
pop_size=10
num_generations = 50;
def matrixgen(size):
    matrix = []
    for i in range(1,size+1):
        matrix.append([i, random.randint(1,10)])
    return matrix


class Population:
    def __init__(self, popind):
        self.pop = np.random.randint(2, size=popind)
    def fitness(self, matrix):
        self.fitness = []
        for i in range(pop_size):
            solmat = np.dot(self.pop[i],matrix)
            if solmat[1]<= max_weight:
                self.fitness.append(solmat[0])
            else:
                self.fitness.append(0)
    #Form of elitist selection
    def selection(self):
        n=len(self.fitness)
        temp = np.array(self.fitness)
        maxindices = (-temp).argsort()[:n]
    def crossover(self, ind1, ind2):
        point = random.randint(0, len(ind1))
        child = ind1.copy()
        for i in range(point,len(ind1)):
            child[i] = ind2[i]
        return child

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    popind = (pop_size, ind_size)
    test = Population(popind)
    a = matrixgen(10)
    test.fitness(a)
    test.selection()
    test.crossover(test.pop[1], test.pop[6])
    print(a)
    print(test.pop)
    print(test.fitness)
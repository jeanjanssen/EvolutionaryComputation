import numpy as np
import random
import math
from matplotlib import pyplot as plt

#TODO write tournament selection, and roulette selection


#EXPERIMENTS
#TODO popsize 25-200, with increases of 25
#TODO probability of crossover and mutation, 0-1, with increases of 0.2
#TODO elitism proportion, 0-10, increases of 2
#TODO number of generations,10k-50k with increases of 10k test



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
    def elitist_selection(self, r=10):

        n=len(self.fitness)
        temp = np.array(self.fitness)
        maxindices = (-temp).argsort()[:n]

        newPop = Population((r, ind_size))
        # best = maxindices[:top]
        i=0
        while len(newPop.pop)<pop_size:

            # crossover
            ind = self.pop[maxindices[i]]
            if i != 0:
                if random.uniform(0, 1) < p_cross:
                    # a = random.randint(0, top-1)
                    a = random.randint(0, len(self.pop)-1)

                    ind = self.single_point_crossover(ind, self.pop[maxindices[a]])
                    # ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

                #mutation
                if random.uniform(0,1) < p_mut:
                    ind = self.single_bit_mutation(ind)

            #add to pop
            newPop.pop = np.append(newPop.pop, [ind], axis=0)
        self.pop = newPop.pop

    # k is tournament size
    def tournament_selection(self, k=5, r=10):

        n = len(self.fitness)
        temp = np.array(self.fitness)
        maxindices = (-temp).argsort()[:n]

        #put random individuals in the new population
        newPop = Population((r, ind_size))
        #add best individual
        newPop.pop = np.append(newPop.pop, [self.pop[maxindices[0]]], axis=0)

        #start tournament selection here
        while len(newPop.pop) < pop_size:
            # tournament
            indexlist=[]
            for i in range (0, k):
                indexlist.append(random.randint(0, n - 1))
            # take best out of k random individuals
            ind =  self.pop[maxindices[min(indexlist)]]
            # crossover
            if random.uniform(0, 1) < p_cross:
                # a = random.randint(0, top-1)
                # tournament
                indexlist = []
                for i in range(0, k):
                    indexlist.append(random.randint(0, n - 1))
                # take best out of k random individuals
                ind2 = self.pop[maxindices[min(indexlist)]]

                ind = self.single_point_crossover(ind, ind2)
                # ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

            # mutation
            if random.uniform(0, 1) < p_mut:
                ind = self.single_bit_mutation(ind)
            newPop.pop = np.append(newPop.pop, [ind], axis=0)
        self.pop = newPop.pop


    def roulette_selection(self, r=10):
        sum_fitness = sum(self.fitness)

        n = len(self.fitness)
        temp = np.array(self.fitness)
        maxindices = (-temp).argsort()[:n]

        # put random individuals in the new population
        newPop = Population((r, ind_size))
        # add best individual
        newPop.pop = np.append(newPop.pop, [self.pop[maxindices[0]]], axis=0)

        prob_sum = 0
        new_prob = []
        for i in range(0,len(self.pop)):
            if self.fitness[i] != 0:
                prob_sum = prob_sum + (self.fitness[i]/sum_fitness)
                new_prob.append(prob_sum)
            else:
                new_prob.append(-1)

        while len(newPop.pop) < pop_size:
            # roulette
            p = random.uniform(0,1)
            for i in range(0, len(new_prob)):
                if new_prob[i]<p:
                    chosen = i
                    break
            ind = self.pop[chosen]

            # crossover
            if random.uniform(0, 1) < p_cross:
                # a = random.randint(0, top-1)
                # roulette
                p = random.uniform(0, 1)
                for i in range(0, len(new_prob)):
                    if new_prob[i] < p:
                        chosen = i
                        break
                ind2 = self.pop[chosen]

                ind = self.single_point_crossover(ind, ind2)
                # ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

            # mutation
            if random.uniform(0, 1) < p_mut:
                ind = self.single_bit_mutation(ind)
            newPop.pop = np.append(newPop.pop, [ind], axis=0)
        self.pop = newPop.pop
    def single_point_crossover(self, ind1, ind2):
        point = random.randint(0, len(ind1))
        child = ind1.copy()
        for i in range(point,len(ind1)):
            child[i] = ind2[i]
        return child
    def two_point_crossover(self, ind1, ind2):
        point1 = random.randint(0, len(ind1))
        point2 = random.randint(0, len(ind1))
        if point1>point2:
            t = point2
            point2 = point1
            point1 = t
        if point1==point2:
            return self.single_point_crossover(ind1,ind2)
        child = ind1.copy()
        for i in range(point1, point2):
            child[i] = ind2[i]
        return child
    def single_bit_mutation(self, ind):
        a = random.randint(0, len(ind)-1)
        if ind[a] == 0:
            ind[a] = 1
        else:
            ind[a] = 0
        return ind
    def calculate_diversity(self):
        self.diversity = []
        for i in range(pop_size):
            div_sum = 0
            for j in range(pop_size):
                div_sum = div_sum + self.distance(self.pop[i], self.pop[j])
            self.diversity.append(div_sum)

        self.diversity_avg = sum(self.diversity)/len(self.diversity)
    def distance(self, ind1, ind2):
        dis = 0
        for i in range(len(ind1)):
            if ind1[i]!=ind2[i]:
                dis = dis + 1
        return dis


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
        test.elitist_selection(r=10)

        # test.tournament_selection(k=5, r=5)
        # test.roulette_selection(r=10)
        test.fitness_calc(a)

        plotlist.append(max(test.fitness))
        #Do avg per hundred
        if i%(0.02*num_generations) == 0:
            plotlist1.append(sum(test.fitness)/len(test.fitness))
            # test.calculate_diversity()
            # plotlist1.append(test.diversity_avg)
            temp= plotlist1[i]
        else:

            plotlist1.append(temp)
        i = i + 1
    print(test.pop)
    print(test.fitness)
    plt.plot(plotlist)
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.plot(plotlist1)
    plt.show()


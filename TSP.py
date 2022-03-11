import numpy as np
import random
from matplotlib import pyplot as plt

#These are standard
pop_size = 50
p_cross = 0.4
p_mut = 0.4
num_generations = 10000

#ALL EXPERIMENTS BELOW
#TODO Test Selection methods, roulette, tournament, elitist
#TODO Randomness proportion selection 0-30, increases of 10
#TODO popsize 25-100, with increases of 25
#TODO probability of crossover and mutation, 0-0.8, with 0, 0.4 and 0.8
#TODO number of generations,1k-10k with increases of 3k per test  (so 1k, 4k, 7k, 10k)
#For all record Maximum fitness
#report on diversity and fitness
#in discussion mention influence for each test on diversity and maximum achieved fitness, also mention how fast the GA converges with a specific test in mind
class Cities:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distancegen(self, city):
        xd = abs(self.x - city.x)
        yd = abs(self.y - city.y)
        d = np.sqrt((xd ** 2) + (yd ** 2))
        return d

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def routegen(cities):
    route = random.sample(cities, len(cities))
    return route


def population(pop_size, cities):
    population = []

    for i in range(0, pop_size):
        population.append(routegen(cities))
    return population


def fitness_calc(route):
    distance = 0
    for i in range(0, len(route)):
        city_a = route[i]
        if i + 1 < len(route):
            city_b = route[i + 1]
        else:
            city_b = route[i - 1]
        distance += city_a.distancegen(city_b)
    fitness = 1 / float(distance)
    return fitness


def elitist_selection(fitness, routes, top=4):
    n = len(fitness)
    temp = np.array(fitness)
    maxindices = (-temp).argsort()[:n]

    newRoutes = []
    # print(routegen(cities))
    a = pop_size - top
    for l in range(0, a):
        newRoutes.append(routegen(cities))
    # for i in range(0, pop_size - top):
    #     newRoutes.append(Cities(x=int(random.random() * 200), y=int(random.random() * 200)))
    i = 0
    while len(newRoutes) < pop_size:

        # crossover
        ind = routes[maxindices[i]]
        if i != 0:

            if random.uniform(0, 1) < p_cross:
                # a = random.randint(0, top-1)
                a = random.randint(0, len(routes) - 1)

                ind = single_point_crossover(ind, routes[maxindices[a]])
                # ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

            # mutation
            if random.uniform(0, 1) < p_mut:
                ind = single_bit_mutation(ind)
            # print("Individual:", ind)
        i = i + 1

        # add to pop
        # newRoutes = np.append(newRoutes, [ind])
        newRoutes.append(ind)
        # print(newRoutes)
        # if i == 2:
        #     print(newRoutes)
        #     exit(0)
    return newRoutes


def tournament_selection(fitness, routes, r=10, k=5):
    n = len(fitness)
    temp = np.array(fitness)
    maxindices = (-temp).argsort()[:n]

    newRoutes = []
    newRoutes.append(routes[maxindices[0]])

    for l in range(0, r):
        newRoutes.append(routegen(cities))
    i = 0
    while len(newRoutes) < pop_size:

        indexlist = []
        for i in range(0, k):
            indexlist.append(random.randint(0, n-1))
        ind = routes[maxindices[min(indexlist)]]

        # crossover

        if random.uniform(0, 1) < p_cross:
            indexlist = []
            for i in range(0, k):
                indexlist.append(random.randint(0, n - 1))
            ind2 = routes[maxindices[min(indexlist)]]

            ind = single_point_crossover(ind, ind2)
            # ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

        # mutation
        if random.uniform(0, 1) < p_mut:
            ind = single_bit_mutation(ind)
        i = i + 1

        # add to pop
        newRoutes.append(ind)

    return newRoutes


def roulette_selection(fitness, routes, r=2):
    sum_fitness = sum(fitness)

    n = len(fitness)
    temp = np.array(fitness)
    maxindices = (-temp).argsort()[:n]

    newRoutes = []
    newRoutes.append(routes[maxindices[0]])

    for l in range(0, r):
        newRoutes.append(routegen(cities))

    prob_sum = 0
    new_prob = []
    for i in range(0, len(fitness)):
        if fitness[i] != 0:
            prob_sum = prob_sum + (fitness[i] / sum_fitness)
            new_prob.append(prob_sum)
        else:
            new_prob.append(-1)

    i = 0
    while len(newRoutes) < pop_size:

        # roulette
        p = random.uniform(0, 1)
        for i in range(0, len(new_prob)):
            if new_prob[i] > p:
                chosen = i
                break
        ind = routes[chosen]

        # crossover
        if random.uniform(0, 1) < p_cross:
            p = random.uniform(0, 1)
            for i in range(0, len(new_prob)):
                if new_prob[i] > p:
                    chosen = i
                    break
            ind2 = routes[chosen]

            ind = single_point_crossover(ind, ind2)
            # ind = self.two_point_crossover(ind, self.pop[maxindices[a]])

        # mutation
        if random.uniform(0, 1) < p_mut:
            ind = single_bit_mutation(ind)
        # print("Individual:", ind)
        i = i + 1

        # add to pop
        newRoutes.append(ind)

    return newRoutes


def single_point_crossover(ind1, ind2):
    point = random.randint(0, len(ind1) - 1)
    child = ind1.copy()
    child.remove(ind2[point])
    child.insert(point, ind2[point])
    return child


def single_bit_mutation(ind):
    point = random.randint(0, len(ind) - 2)
    test = ind.copy()
    tempind = test[point]
    test[point] = test[point + 1]
    test[point + 1] = tempind
    return test


if __name__ == '__main__':
    cities = []
    random.seed(0)
    for i in range(0, 10):
        cities.append(Cities(x=int(random.random() * 200), y=int(random.random() * 200)))
    # print("Cities =", cities)
    routes = []
    fitness = []
    for i in range(0, pop_size):
        routes.append(routegen(cities))
        fitness.append(fitness_calc(routes[i]))
    # print("Route1 =", routes[1])
    i = 0
    plotlist = []
    plotlist1 = []
    a=0
    index=0
    while i < num_generations:
        # print(i)

        # routes = elitist_selection(fitness, routes, top=4)
        routes = tournament_selection(fitness, routes, r=10)
        # routes = roulette_selection(fitness, routes)
        fitness = []
        for k in range(0, len(routes)):
            fitness.append(fitness_calc(routes[k]))
        plotlist.append(max(fitness))

        if i%(0.02*num_generations) == 0:
            # plotlist1.append(sum(test.fitness)/len(test.fitness))
            plotlist1.append(sum(fitness) / len(fitness))
            temp= plotlist1[i]
        else:

            plotlist1.append(temp)
        i = i + 1
    print("Maximum achieved fitness: ", max(plotlist1))
    plt.plot(plotlist)
    plt.plot(plotlist1)
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.show()

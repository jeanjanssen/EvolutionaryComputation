import numpy as np
import random
from matplotlib import pyplot as plt

pop_size = 100
p_cross = 0.4
p_mut = 0.4
num_generations = 2000


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


def tournament_selection(fitness, routes, r=5, k=5):
    n = len(fitness)
    temp = np.array(fitness)
    maxindices = (-temp).argsort()[:n]

    newRoutes = []
    newRoutes.append(routes[maxindices[0]])
    # print(newRoutes)
    # exit(0)
    # print(routegen(cities))
    for l in range(0, r):
        newRoutes.append(routegen(cities))
    # for i in range(0, pop_size - top):
    #     newRoutes.append(Cities(x=int(random.random() * 200), y=int(random.random() * 200)))
    i = 0
    while len(newRoutes) < pop_size:

        indexlist = []
        for i in range(0, k):
            indexlist.append(random.randint(0, n-1))
        ind = routes[maxindices[min(indexlist)]]

        # crossover
        # ind = routes[maxindices[i]]

        if random.uniform(0, 1) < p_cross:
            # a = random.randint(0, top-1)
            # a = random.randint(0, len(routes) - 1)

            indexlist = []
            for i in range(0, k):
                indexlist.append(random.randint(0, n - 1))
            ind2 = routes[maxindices[min(indexlist)]]

            ind = single_point_crossover(ind, ind2)
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


def roulette_selection(fitness, routes, top=4):
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


def single_point_crossover(ind1, ind2):
    point = random.randint(0, len(ind1) - 1)
    child = ind1.copy()
    child.remove(ind2[point])
    child.insert(point, ind2[point])
    return child


def single_bit_mutation(ind):
    point = random.randint(0, len(ind) - 2)
    tempind = ind[point]
    ind[point] = ind[point + 1]
    ind[point + 1] = tempind
    return ind


if __name__ == '__main__':
    cities = []

    for i in range(0, pop_size):
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
    while i < num_generations:
        print(i)
        #elitist_selection(fitness, routes, top=4)
        tournament_selection(fitness, routes)
        tempfitness = []
        for k in range(len(routes)):
            f = fitness_calc(routes[k])
            tempfitness.append(f)
        fitness = tempfitness
        plotlist.append(max(fitness))
        plotlist1.append(sum(fitness) / len(fitness))
        i = i + 1
    plt.plot(plotlist)
    #plt.plot(plotlist1)
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.show()

import numpy as np
import random

pop_size = 10


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


if __name__ == '__main__':
    cities = []

    for i in range(0, 10):
        cities.append(Cities(x=int(random.random() * 200), y=int(random.random() * 200)))
    print(cities)

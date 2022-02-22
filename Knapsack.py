import numpy as np
import math


class Knapsack:
    index = np.arange(1, 10)
    weight = np.randint(1, 10)
    max_weight = 25
    value = index

    solutions = 8
    pop_size = (solutions, index.shape[0])

    def gen_pop(pop_size):
        initial_pop = np.random.randint(2, size=pop_size)
        initial_pop = initial_pop.astype(int)
        return initial_pop

    num_generations = 50;


if __name__ == '__main__':
    print("Hello World!")

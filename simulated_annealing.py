import random
import numpy as np
import copy

class SimulatedAnnealing:
    def __init__(self, problem, initial_temperature = 1000, min_temperature = 0.01,
                 max_iteration = 100, steps = 50, cooling_parameter = 0.98):

        self.init_temp = initial_temperature
        self.problem = problem
        self.min_temp = min_temperature
        self.max_iter = max_iteration
        self.steps = steps
        self.cool = cooling_parameter

    def evaluate(self, target):
        return self.problem.decode(target[0], target[1])

    def random_swap(self, target):
        new_target = copy.deepcopy(target)
        ops = list(new_target[1])
        j, _ = random.choice(ops)
        base = random.random()
        for o in range(len(self.problem.jobs_data[j])):
            new_target[1][(j,o)] = base + 0.1 * o

        return new_target

    def run(self, seed_solution):
        t = self.init_temp
        k = 0
        curr = seed_solution
        curr_obj = self.evaluate(curr)
        best = curr
        best_obj = curr_obj

        while t > self.min_temp and k < self.max_iter:
            for _ in range(self.steps):
                new = self.random_swap(curr)
                new_obj = self.evaluate(new)
                diff = new_obj - curr_obj
                prob = min(1, np.exp(-diff / t))

                if prob >= random.uniform(0, 1):
                    curr = new
                    curr_obj = new_obj

                if new_obj < best_obj:
                    best, best_obj = new, new_obj

            t = t * self.cool
            k += 1

        return best_obj, best
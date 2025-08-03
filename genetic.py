import random
import copy
from collections import defaultdict, deque

class GeneticAlgorithm:
    def __init__(self, problem, pop_size = 100, max_generations = 50, cross_p = 0.8, mut_p = 0.2):
        self.problem = problem
        self.pop_size = pop_size
        self.gen_max = max_generations
        self.cross_p = cross_p
        self.mut_p = mut_p
        self.population = []

    def init_population(self, seed_solution):
        self.population.append(seed_solution)
        for _ in range(self.pop_size - 1):
            if random.random() < self.mut_p:
                self.population.append(self.problem.encode())
            else:
                self.population.append(self.mutate(seed_solution))

    def evaluate_fitness(self, target):
        return self.problem.decode(target[1])

    def mutate(self, target):
        new_target = copy.deepcopy(target)
        ops = list(new_target[1])

        if random.random() < self.mut_p:
            j,o = random.choice(ops)
            centers = self.problem.jobs_data[j][o][1]
            p = random.choice([centers])
            k = random.randrange(self.problem.center_caps[p])
            new_target[0][(j,o)] = f"{p}_{k}"

        if random.random() < self.mut_p:
            new_assign, new_times = new_target
            schedule = self.problem.generate_schedule(new_assign, new_times)
            by_machine = defaultdict(list)
            for j, ops in schedule.items():
                for o, m, st, end in ops:
                    by_machine[m].append((st, j, o, end))

            machines = [m for m, lst in by_machine.items() if len(lst) > 1]
            if not machines:
                return new_assign, new_times

            m = random.choice(machines)
            ops = sorted(by_machine[m], key = lambda x: x[0])
            idx = random.randint(1, len(ops) - 1)
            st_cur, j_cur, o_cur, end_cur = ops[idx]
            st_pre, j_pre, o_pre, end_pre = ops[idx - 1]
            j_temp, o_temp = None, None
            if idx >= 2:
                _, j_temp, o_temp, _ = ops[idx - 2]

            if j_cur != j_pre:
                if j_temp:
                    new_times[(j_cur, o_cur)] = (st_pre + self.problem.setup_times[m[:2]][((j_temp, o_temp), (j_cur, o_cur))], st_pre + end_cur - st_cur)
                else:
                    new_times[(j_cur, o_cur)] = (st_pre, st_pre + end_cur - st_cur)

                s = new_times[(j_cur, o_cur)][1] + self.problem.setup_times[m[:2]][((j_cur, o_cur), (j_pre, o_pre))]
                new_times[(j_pre, o_pre)] = (s, s + end_pre - st_pre)

        return new_target

    def linear_order_crossover(self, parent1, parent2):
        parent1_times, parent2_times = parent1[1], parent2[1]
        parent1_ops, parent2_ops = [], []

        for (j, o), (st, end) in parent1_times.items():
            parent1_ops.append((j, o, st))

        parent1_ops.sort(key = lambda x: x[2])

        for (j, o), (st, end) in parent2_times.items():
            parent2_ops.append((j, o, st))

        parent2_ops.sort(key = lambda x: x[2])

        size = len(parent1_ops)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size

        # Step 1: 拿一段排序片段
        for i in range(start, end + 1):
            child[i] = (parent1_ops[i][0], parent1_ops[i][1])

        used_ops = [(j,o) for (j, o) in child[start:end + 1]]
        # Step 2: 剩餘由parent2補完，保持原順序
        idx = 0
        for (j, o, st) in parent2_ops:
            if (j,o) not in used_ops:
                while start <= idx <= end:
                    idx += 1
                if idx < size:
                    child[idx] = (j, o)
                    idx += 1

        groups = defaultdict(list)
        for (x, y) in child:
            groups[x].append((x, y))

        for x in groups:
            groups[x].sort(key = lambda t: t[1])
            groups[x] = deque(groups[x])

        child_priority = []
        for x, _ in child:
            child_priority.append(groups[x].popleft())

        child_machine, child_times = self.problem.encode(priority = child_priority)
        res = [child_machine, child_times]
        res = self.mutate(res)

        return res

    def roulette_selection(self, number = 2):
        fitness_values = [1.0 / (self.evaluate_fitness(target) + 1e-6) for target in self.population]
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        cumulative_probs = []
        cumulative = 0.0
        for p in probabilities:
            cumulative += p
            cumulative_probs.append(cumulative)

        res = []
        for _ in range(number):
            spin = random.random()
            for i, cp in enumerate(cumulative_probs):
                if spin <= cp:
                    res.append(self.population[i])
                    break
        return res

    def run(self, seed_solution):
        self.init_population(seed_solution)
        best = None
        best_obj = float('inf')

        for _ in range(self.gen_max):
            for target in self.population:
                obj = self.evaluate_fitness(target)
                if obj < best_obj:
                    best_obj, best = obj, target

            next_pop = []
            while len(next_pop) < self.pop_size:
                p1, p2 = self.roulette_selection()
                next_pop.extend([p1, p2])

                if random.random() < self.cross_p:
                    child = self.linear_order_crossover(p1, p2)
                    next_pop.append(child)

            self.population = next_pop[:self.pop_size]

        return best_obj, best


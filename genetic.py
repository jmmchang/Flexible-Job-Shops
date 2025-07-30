import random
import copy
from functions import decode_schedule
from collections import defaultdict

class GeneticAlgorithm:
    def __init__(self, problem, pop_size = 100, max_generations = 50, cross_p = 0.8, mut_p = 0.2):
        self.problem = problem
        self.pop_size = pop_size
        self.gen_max = max_generations
        self.cross_p = cross_p
        self.mut_p = mut_p
        self.population = []

    def init_population(self, seed_solution):
        self.population.append(self.problem.encode(seed_solution))
        for _ in range(self.pop_size - 1):
            self.population.append(self.problem.encode())

    def evaluate_fitness(self, target):
        fitness = self.problem.decode(target[0], target[1])
        return fitness

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
            new_assign, new_prio = new_target
            schedule = decode_schedule(
                self.problem.jobs_data,
                self.problem.release,
                self.problem.setup_times,
                new_assign,
                new_prio)

            by_machine = defaultdict(list)
            for j, ops in schedule.items():
                for o, m, st, _ in ops:
                    by_machine[m].append((st, j, o))

            # 2. 隨機挑一台有 >=2 道工序的機台
            machines = [m for m, lst in by_machine.items() if len(lst) > 1]
            if not machines:
                return new_assign, new_prio
            m = random.choice(machines)

            # 3. 排序後隨機選一個「當前」索引 idx，與 idx-1 的工序做交換
            ops = sorted(by_machine[m], key=lambda x: x[0])
            idx = random.randint(1, len(ops) - 1)
            _, j_cur, o_cur = ops[idx]
            _, j_pre, o_pre = ops[idx - 1]
            if j_cur != j_pre:
                new_prio[(j_cur, o_cur)], new_prio[(j_pre, o_pre)] = new_prio[(j_pre, o_pre)], new_prio[(j_cur, o_cur)]

            for j in (j_pre, j_cur):
                ops_count = len(self.problem.jobs_data[j])
                # 收集所有 op 的 priority，排序
                sorted_vals = sorted(new_prio[(j, o)] for o in range(ops_count))
                # 依序寫回，保證 op0 < op1 < ...
                for o, val in enumerate(sorted_vals):
                    new_prio[(j, o)] = val

        return new_target

    def linear_order_crossover(self, parent1, parent2):
        jobs = list(self.problem.jobs_data.keys())
        size = len(jobs)
        start, end = sorted(random.sample(range(size), 2))

        # Step 1: 拿一段 job base 排序片段
        segment_jobs = jobs[start:end+1]
        segment_bases = {j: parent1[1][(j, 0)] for j in segment_jobs}

        # Step 2: 剩餘 job 由 parent2 補完，保持原順序
        remainder_jobs = [j for j in jobs if j not in segment_jobs]
        remainder_bases = {j: parent2[1][(j, 0)] for j in remainder_jobs}

        # Step 3: 合併 base 順序
        job_order = remainder_jobs[:start] + segment_jobs + remainder_jobs[start:]
        combined_bases = {**remainder_bases, **segment_bases}

        # Step 4: 為每個 job 建立 priority[(j,o)]，保工序順序
        child_priority = {}
        for j in job_order:
            base = combined_bases[j]
            for o in range(len(self.problem.jobs_data[j])):
                child_priority[(j, o)] = base + 0.1 * o

        child = (parent1[0], child_priority)
        child = self.mutate(child)

        return child

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

                if random.random() < self.cross_p:
                    c1 = self.linear_order_crossover(p1, p2)
                    c2 = self.linear_order_crossover(p1, p2)
                else:
                    c1, c2 = p1, p2

                next_pop.extend([c1, c2])

            self.population = next_pop[:self.pop_size]

        return best_obj, best


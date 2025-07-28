import random

class SchedulingProblem:
    def __init__(self, jobs_data, release_dates, due_dates,
                 weights, setup_times, center_caps):
        self.jobs_data = jobs_data
        self.release = release_dates
        self.due_dates = due_dates
        self.weights = weights
        self.setup_times = setup_times
        self.center_caps = center_caps

    def encode(self, seed_solution = None):
        machine_assign = {}
        priorities = {}

        if seed_solution:
            ma_seed, pr_seed = seed_solution
            machine_assign = ma_seed.copy()
            priorities     = pr_seed.copy()
        else:
            for j, ops in self.jobs_data.items():
                base = random.random()
                for o in range(len(ops)):
                    centers = self.jobs_data[j][o][1]
                    p = random.choice([centers])
                    k = random.randrange(self.center_caps[p])
                    machine_assign[(j,o)] = f"{p}_{k}"
                    priorities[(j,o)] = base + 0.001 * o

        return machine_assign, priorities

    def decode(self, machine_assign, priorities):
        machines = []

        for c, cap in self.center_caps.items():
            for k in range(cap):
                machines.append(f"{c}_{k}")

        mc_timeline = {m:[] for m in machines}
        doc_times = {}
        ops_sorted = sorted(priorities.items(), key = lambda x: x[1])

        for (j, o), _ in ops_sorted:
            m = machine_assign[(j,o)]
            p = m.split('_')[0]
            duration, _ = self.jobs_data[j][o]

            t = self.release[j]
            if o > 0:
                prev_end = doc_times[(j, o-1)][1]
                t = max(t, prev_end)

            timeline = sorted(mc_timeline[m], key = lambda x: x[0])
            for prev_end, prev_op in timeline:
                s = self.setup_times.get(p, {}).get((prev_op, (j,o)), 0)
                t = max(t, prev_end + s)

            start, end = t, t + duration
            mc_timeline[m].append((end, (j,o)))
            doc_times[(j,o)] = (start, end)

        obj = 0
        for j, ops in self.jobs_data.items():
            last_endtime = doc_times[(j, len(ops)-1)][1]
            obj += self.weights[j] * max(0, last_endtime - self.due_dates[j])

        return obj

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
        new_target = target
        ops = list(new_target[1])
        if random.random() < self.mut_p:
            j,o = random.choice(ops)
            centers = self.problem.jobs_data[j][o][1]
            p = random.choice([centers])
            k = random.randrange(self.problem.center_caps[p])
            new_target[0][(j,o)] = f"{p}_{k}"

        if random.random() < self.mut_p:
            j, _ = random.choice(ops)
            base = random.random()
            for o in range(len(self.problem.jobs_data[j])):
                new_target[1][(j,o)] = base + 0.001 * o

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
                child_priority[(j, o)] = base + 0.001 * o

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

def decode_schedule(jobs_data,
                    release_dates,
                    setup_times,
                    machine_assign,
                    priorities):
    """
    將 priority-based GA 染色體轉為可視甘特圖的 schedule。
    Arguments:
      jobs_data: {j: [(duration, [centers]), ...]}
      release_dates: {j: rj}
      setup_times: {p: {((j1,o1),(j2,o2)): s_time}}
      machine_assign: {(j,o): "C1_0", ...}
      priorities: {(j,o): float, ...}

    Returns:
      schedule: { j: [ (o, m, start, end), ... ] }
    """

    # 1. 準備：各機台的時間表
    machines = set(machine_assign.values())
    mc_timeline = {m: [] for m in machines}
    so_times = {}

    # 2. 按 priority 排序所有 (j,o)
    ops = list(priorities.items())
    # 如果 priority 相同，用 (j,o) 做小 tiebreaker
    ops_sorted = sorted(ops, key=lambda x: (x[1], x[0]))

    # 3. 逐一排程
    for (j,o), pr in ops_sorted:
        m = machine_assign[(j,o)]
        # 3.1 計算起始下界 t0
        t0 = release_dates.get(j, 0)
        if o > 0:
            # 前序完成
            prev_en = so_times[(j, o-1)][1]
            t0 = max(t0, prev_en)

        # 3.2 機台上考慮換線與已排工序
        timeline = sorted(mc_timeline[m], key=lambda x: x[0])  # (end_prev,(j_prev,o_prev))
        t = t0
        center = m.split('_')[0]

        for end_prev, prev_op in timeline:
            # 從 prev_op → current op 的換線
            s = setup_times.get(center, {}).get((prev_op, (j,o)), 0)
            # 若 t < end_prev + s，必須推到 end_prev+s
            if t < end_prev + s:
                t = end_prev + s

        # 3.3 計算結束時間並更新
        st, en = t, t + jobs_data[j][o][0]
        so_times[(j,o)] = (st, en)
        mc_timeline[m].append((en, (j,o)))

    # 4. 建立 schedule dict
    schedule = {}
    for (j,o), (st,en) in so_times.items():
        m = machine_assign[(j,o)]
        schedule.setdefault(j, []).append((o, m, st, en))

    return schedule
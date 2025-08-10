import random
import copy
from collections import defaultdict, deque

class GeneticAlgorithm:
    """
        A template for a Genetic Algorithm applied to a scheduling problem.

        Attributes:
            problem:    problem interface with methods:
                            - encode(assign=None, priority=None) -> (machine_assign, time_dict)
                            - decode(time_dict) -> objective_value
                            - generate_schedule(assign, time_dict) -> detailed schedule
            pop_size:   Population size
            gen_max:    Maximum number of generations
            cross_p:    Crossover probability
            mut_p:      Probability of mutation
    """

    def __init__(self, problem, pop_size = 100, max_generations = 50, cross_p = 0.8, mut_p = 0.2):
        self.problem = problem
        self.pop_size = pop_size
        self.gen_max = max_generations
        self.cross_p = cross_p
        self.mut_p = mut_p
        self.population = []

    def init_population(self, seed_solution):
        """
        Initialize the population using a seed solution.
        """

        self.population.append(seed_solution)
        for _ in range(self.pop_size - 1):
            if random.random() < self.mut_p:
                self.population.append(self.problem.encode())
            else:
                self.population.append(self.mutate(seed_solution))

    def evaluate_fitness(self, target):
        """
        Compute the fitness (objective) of an individual.
        """

        return self.problem.decode(target[1])

    def _mutate_assignment(self, individual):
        """
        Single-point machine assignment mutation:
        """

        new_assign, new_times = self.problem.encode()

        return new_assign, new_times

    def _mutate_sequence(self, individual):
        """
        Sequence mutation: pick a machine with ≥2 operations,
        sort its ops by start time, swap a random adjacent pair,
        then re-encode using the new job-operation priority.
        """

        assign, times = individual

        # 1. Build full schedule to inspect start times
        schedule = self.problem.generate_schedule(assign, times)

        # 2. Group operations by machine
        by_machine = defaultdict(list)
        for job_id, ops in schedule.items():
            for op_idx, m, st, _ in ops:
                by_machine[m].append((job_id, op_idx, st))

        # 3. Find machines with at least two ops
        candidates = [m for m, lst in by_machine.items() if len(lst) > 1]
        if not candidates:
            return assign, times

        # 4. Choose one machine and swap a random adjacent pair in time order
        m = random.choice(candidates)
        ops_m = sorted(by_machine[m], key=lambda x: x[2])
        i = random.randint(1, len(ops_m) - 1)
        # Swap positions in the time-order list
        ops_m[i - 1], ops_m[i] = ops_m[i], ops_m[i - 1]

        # 5. Rebuild a global priority list, preserving job-internal order
        priority_list = []
        temp = []
        for machine_ops in by_machine.values():
            temp.extend(machine_ops)

        # Replace the sequence for machine m
        temp = [x for x in temp if x[0] != m] + ops_m
        temp.sort(key = lambda x: x[2])
        for job_id, op_idx, _ in temp:
            priority_list.append((job_id, op_idx))

        # 6. Encode again to regenerate machine_assign & times
        new_assign, new_times = self.problem.encode(priority = priority_list)

        return new_assign, new_times

    def mutate(self, individual):
        """
        Top-level mutation: deep-copy an individual and apply
        assignment- or sequence-based mutations based on probabilities.
        """

        new_ind = copy.deepcopy(individual)

        # Machine-assignment mutation
        if random.random() < self.mut_p:
            new_ind = self._mutate_assignment(new_ind)

        # Sequence (order) mutation on a single machine
        if random.random() < self.mut_p:
            new_ind = self._mutate_sequence(new_ind)

        return new_ind

    def linear_order_crossover(self, parent1, parent2):
        """
        Order-based crossover:
        1. Extract operation sequences sorted by start times from both parents.
        2. Copy a random slice from parent1 into the child.
        3. Fill the remaining positions from parent2 in order.
        4. Reconstruct job-internal priority and re-encode.
        5. Apply mutation to the child.
        """

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

        for i in range(start, end + 1):
            child[i] = (parent1_ops[i][0], parent1_ops[i][1])

        used_ops = [(j,o) for (j, o) in child[start:end + 1]]
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
        """
          Roulette-wheel selection: pick k individuals proportional to their fitness.
        """

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
        """
        Execute the GA:
        1. Initialize population
        2. Iterate for gen_max generations:
           a. Evaluate fitness, track best
           b. Perform elitism, selection, crossover, mutation
        3. Return best objective and solution found
        """

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


import itertools
from ortools.sat.python import cp_model
import random
import numpy as np
import matplotlib.pyplot as plt

def solve_fjs_with_parallel_machines(jobs_data,
                                     release_dates,
                                     due_dates,
                                     weights,
                                     setup_times,
                                     center_caps):
    model = cp_model.CpModel()

    # 1. 生成實體機台
    machines, center_of = [], {}
    for p, cap in center_caps.items():
        for k in range(cap):
            m = f"{p}_{k}"
            machines.append(m)
            center_of[m] = p

    # 2. 決策變數
    horizon = sum(d for ops in jobs_data.values() for d,_ in ops) + max(due_dates.values())
    start, end, assign, tardiness = {}, {}, {}, {}

    for j, ops in jobs_data.items():
        for o, (dur, centers) in enumerate(ops):
            start[j, o] = model.NewIntVar(0, horizon, f"s_j{j}o{o}")
            end[j, o]   = model.NewIntVar(0, horizon, f"e_j{j}o{o}")
            # 只為允許的機台建 assign 變數
            valid = [m for m in machines if center_of[m] in centers]
            bools = []
            for m in valid:
                b = model.NewBoolVar(f"a_j{j}o{o}_m{m}")
                assign[j, o, m] = b
                bools.append(b)
                model.Add(end[j, o] == start[j, o] + dur).OnlyEnforceIf(b)

            model.AddExactlyOne(bools)

    # 3. 釋放時間與工序順序
    for j, ops in jobs_data.items():
        model.Add(start[j, 0] >= release_dates[j])
        for o in range(len(ops) - 1):
            model.Add(start[j, o + 1] >= end[j, o])

    # 4. 機台互斥 + 序依換線
    for m in machines:
        tasks = [(j, o) for j, ops in jobs_data.items()
                        for o, (_, centers) in enumerate(ops)
                        if center_of[m] in centers]
        for (j1, o1), (j2, o2) in itertools.combinations(tasks, 2):
            b = model.NewBoolVar(f"ord_{j1}o{o1}_{j2}o{o2}_on_{m}")
            p = center_of[m]
            s12 = setup_times.get(p, {}).get(((j1, o1), (j2, o2)), 0)
            s21 = setup_times.get(p, {}).get(((j2, o2), (j1, o1)), 0)

            model.Add(start[j2, o2] >= end[j1, o1] + s12)\
                 .OnlyEnforceIf([assign[j1, o1, m],
                                 assign[j2, o2, m], b])
            model.Add(start[j1, o1] >= end[j2, o2] + s21)\
                 .OnlyEnforceIf([assign[j1, o1, m],
                                 assign[j2, o2, m], b.Not()])

    # 5. 拖期與目標
    for j, ops in jobs_data.items():
        last = len(ops) - 1
        tardiness[j] = model.NewIntVar(0, horizon, f"T_j{j}")
        model.Add(tardiness[j] >= end[j, last] - due_dates[j])

    model.Minimize(sum(weights[j] * tardiness[j] for j in jobs_data))

    # 6. 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for j, ops in jobs_data.items():
            schedule[j] = []
            for o in range(len(ops)):
                st = solver.Value(start[j, o])
                en = solver.Value(end[j, o])
                # 只在 assign 有定義的機台中找被選中的那一台
                m_assigned = next(
                    m for m in machines
                    if (j, o, m) in assign
                    and solver.Value(assign[j, o, m]) == 1
                )
                schedule[j].append((o, m_assigned, st, en))
        return schedule, solver.ObjectiveValue()

    return None, None

def generate_random_instance(num_jobs = 15, centers = ('C1','C2','C3',"C4"),
                             center_caps = {'C1':3,'C2':2,'C3':3,"C4":1},
                             num_ops = {'C1':4,'C2':3,'C3':2,"C4":1}):

    jobs_data, release_dates, due_dates, weights = {}, {}, {}, {}
    setup_times = {p:{} for p in centers}

    for j in range(num_jobs):
        ops = []
        total_dur = 0
        for c in centers:
            for _ in range(num_ops[c]):
                dur = random.randint(20, 50)
                total_dur += dur
                ops.append((dur, c))

        jobs_data[j] = ops
        due_dates[j]     = random.randint(np.floor(total_dur * 2), np.floor(total_dur * 4))
        release_dates[j] = random.randint(0, due_dates[j] // 2)
        weights[j]       = random.randint(1, 5)

    # 隨機設定序依換線時間
    for p in centers:
        all_ops = [(j, o) for j, ops in jobs_data.items()
                          for o in range(len(ops))
                          if p in ops[o][1]]
        for (j1,o1),(j2,o2) in itertools.permutations(all_ops, 2):
            setup_times[p][((j1,o1),(j2,o2))] = random.randint(1, 10)

    return jobs_data, release_dates, due_dates, weights, setup_times, center_caps

def plot_gantt(schedule, title="Gantt Chart"):
    # 1. 準備 y 軸：機台列表與對應座標
    machines = sorted({ m for ops in schedule.values() for _, m, _, _ in ops })
    y_pos = { m: i for i, m in enumerate(machines) }

    # 2. 顏色映射 (以 job_id 分組上色)
    cmap = plt.get_cmap("tab20")
    job_ids = sorted(schedule)
    color_map = { j: cmap(i % 20) for i, j in enumerate(job_ids) }

    # 3. 繪圖
    fig, ax = plt.subplots(figsize=(12, len(machines)*0.6+1))
    for j, ops in schedule.items():
        for op_idx, m, st, en in ops:
            ax.barh(y_pos[m],
                    en - st,
                    left=st,
                    height=0.4,
                    color=color_map[j],
                    edgecolor="black")
            ax.text(st + (en-st)/2,
                    y_pos[m],
                    f"J{j}O{op_idx}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8)

    # 4. 美化
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

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
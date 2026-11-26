"""
Project 10: Smart Dependency and Project Execution Manager
-----------------------------------------------------------
Tasks  = vertices
Edge   = directed edge  A -> B  means "A must finish before B starts"

Implements:
  1. Topological sort using BFS (Kahn's Algorithm)
  2. Topological sort using DFS (3-colour method)
  3. Cycle detection (with the actual cycle path)
  4. Level-wise schedule (tasks that can run in parallel)
  5. Performance comparison of both approaches
"""

import random
import time
from collections import defaultdict, deque


class ProjectGraph:
    def __init__(self):
        self.adj = defaultdict(list)   # task -> list of dependent tasks
        self.tasks = []                # keeps insertion order
        self._seen = set()

    # ---------- graph building ----------
    def add_task(self, task):
        if task not in self._seen:
            self._seen.add(task)
            self.tasks.append(task)
            self.adj[task]  # make sure key exists

    def add_dependency(self, before, after):
        """'before' must be completed before 'after'."""
        self.add_task(before)
        self.add_task(after)
        self.adj[before].append(after)

    # ---------- 1. Kahn's Algorithm (BFS) ----------
    def topo_kahn(self):
        indeg = {t: 0 for t in self.tasks}
        for u in self.tasks:
            for v in self.adj[u]:
                indeg[v] += 1

        queue = deque(t for t in self.tasks if indeg[t] == 0)
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(order) != len(self.tasks):
            return None          # cycle exists
        return order

    # ---------- level-wise (parallel) schedule using Kahn ----------
    def parallel_schedule(self):
        indeg = {t: 0 for t in self.tasks}
        for u in self.tasks:
            for v in self.adj[u]:
                indeg[v] += 1

        current = [t for t in self.tasks if indeg[t] == 0]
        levels, done = [], 0
        while current:
            levels.append(current)
            done += len(current)
            nxt = []
            for u in current:
                for v in self.adj[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        nxt.append(v)
            current = nxt
        return levels if done == len(self.tasks) else None

    # ---------- 2. DFS based topological sort + cycle detection ----------
    # colours: 0 = WHITE (unvisited), 1 = GRAY (in current path), 2 = BLACK (done)
    def topo_dfs(self):
        """Returns (order, cycle). If a cycle is found, order is None."""
        colour = {t: 0 for t in self.tasks}
        parent = {}
        post = []

        for start in self.tasks:
            if colour[start] != 0:
                continue
            colour[start] = 1
            stack = [(start, iter(self.adj[start]))]   # iterative DFS (no recursion limit)
            while stack:
                u, it = stack[-1]
                advanced = False
                for v in it:
                    if colour[v] == 0:
                        colour[v] = 1
                        parent[v] = u
                        stack.append((v, iter(self.adj[v])))
                        advanced = True
                        break
                    elif colour[v] == 1:               # back edge -> cycle
                        cycle = [u]
                        x = u
                        while x != v:
                            x = parent[x]
                            cycle.append(x)
                        cycle.reverse()
                        return None, cycle
                if not advanced:
                    colour[u] = 2
                    post.append(u)
                    stack.pop()

        post.reverse()
        return post, None

    # ---------- 3. Cycle detection ----------
    def find_cycle(self):
        _, cycle = self.topo_dfs()
        return cycle

    # ---------- validation helper ----------
    def is_valid_order(self, order):
        pos = {t: i for i, t in enumerate(order)}
        return all(pos[u] < pos[v] for u in self.tasks for v in self.adj[u])


# =====================================================================
# Performance analysis
# =====================================================================
def random_dag(n, edges_per_node=3, seed=1):
    rnd = random.Random(seed)
    ranks = list(range(n))
    rnd.shuffle(ranks)                       # hide the true order
    g = ProjectGraph()
    for i in range(n):
        g.add_task(i)
    for _ in range(n * edges_per_node):
        a, b = rnd.sample(range(n), 2)
        if ranks[a] < ranks[b]:
            g.add_dependency(a, b)
        else:
            g.add_dependency(b, a)
    return g


def timeit(fn, runs=3):
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000  # ms


def benchmark(sizes=(1000, 5000, 10000, 50000, 100000)):
    print(f"\n{'V':>8} {'E':>9} {'Kahn (ms)':>12} {'DFS (ms)':>12}")
    print("-" * 44)
    for n in sizes:
        g = random_dag(n)
        e = sum(len(g.adj[u]) for u in g.tasks)
        tk = timeit(g.topo_kahn)
        td = timeit(g.topo_dfs)
        print(f"{n:>8} {e:>9} {tk:>12.2f} {td:>12.2f}")


# =====================================================================
# Demo
# =====================================================================
def demo():
    g = ProjectGraph()
    deps = [
        ("Requirements", "Design"),
        ("Design", "Frontend"),
        ("Design", "Backend"),
        ("Backend", "Database Setup"),
        ("Database Setup", "Integration"),
        ("Frontend", "Integration"),
        ("Integration", "Testing"),
        ("Testing", "Deployment"),
    ]
    for a, b in deps:
        g.add_dependency(a, b)

    print("=== Valid project (DAG) ===")
    k = g.topo_kahn()
    d, _ = g.topo_dfs()
    print("Kahn (BFS) order :", " -> ".join(k), "| valid:", g.is_valid_order(k))
    print("DFS order        :", " -> ".join(d), "| valid:", g.is_valid_order(d))
    print("\nParallel schedule (level-wise):")
    for i, lvl in enumerate(g.parallel_schedule(), 1):
        print(f"  Step {i}: {', '.join(lvl)}")

    print("\n=== Project with circular dependency ===")
    g.add_dependency("Deployment", "Design")   # creates a cycle
    print("Kahn result      :", g.topo_kahn())
    cycle = g.find_cycle()
    print("Cycle detected   :", " -> ".join(cycle + [cycle[0]]))

    print("\n=== Performance analysis (random DAGs, E = 3V) ===")
    benchmark()


if __name__ == "__main__":
    demo()

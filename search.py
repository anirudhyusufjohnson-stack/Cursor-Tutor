"""Uninformed search: Breadth-First, Depth-First and Uniform-Cost Search over an app's screen graph.

An app is a graph. A STATE is a screen ("chat_open"), an ACTION is something the user does
("Click the paperclip"), and each action has a COST (the effort for a beginner). A how-to answer is a
path from the screen the user is on (the start state) to the screen where the goal is done (the goal state).

  BFS  expands the shallowest screens first            -> fewest steps (optimal when every step costs the same)
  DFS  follows one branch as deep as it can first      -> finds some route, often not the best, uses little memory
  UCS  expands the cheapest path so far first          -> lowest total effort (optimal for any non-negative costs)

Each search returns how many screens it expanded, so the algorithms can be compared."""
import heapq
import itertools
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Flow:
    app: str
    name: str
    start: str
    edges: dict                  # state -> [(next_state, action, cost), ...]  in the order they are listed
    goals: list                  # [{"guide": title, "state": goal_state}]

    def states(self):
        seen = list(self.edges)
        for outs in self.edges.values():
            for to, _a, _c in outs:
                if to not in seen:
                    seen.append(to)
        return seen


@dataclass
class Plan:
    algorithm: str
    found: bool
    states: list = field(default_factory=list)       # screens from start to goal
    actions: list = field(default_factory=list)      # what to do to get from each screen to the next
    cost: float = 0.0
    expanded: int = 0                                # screens taken off the frontier and examined
    generated: int = 0                               # screens put on the frontier
    max_frontier: int = 0                            # most screens waiting at once (memory)
    order: list = field(default_factory=list)        # the order screens were expanded in

    @property
    def steps(self):
        return len(self.actions)


def load_flows(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    flows = {}
    for f in data.get("flows", []):
        edges = {}
        for a, b, action, cost in f["edges"]:
            edges.setdefault(a, []).append((b, action, float(cost)))
        flows[f["app"]] = Flow(f["app"], f["name"], f["start"], edges, f.get("goals", []))
    return flows


def _build(algorithm, found, goal, parent, expanded, generated, max_frontier, order):
    if not found:
        return Plan(algorithm, False, expanded=expanded, generated=generated, max_frontier=max_frontier, order=order)
    states, actions, cost, cur = [goal], [], 0.0, goal
    while parent[cur] is not None:
        prev, action, c = parent[cur]
        states.append(prev)
        actions.append(action)
        cost += c
        cur = prev
    return Plan(algorithm, True, states[::-1], actions[::-1], cost, expanded, generated, max_frontier, order)


def bfs(flow, start, goal):
    """Breadth-first graph search (goal test when a screen is first reached)."""
    parent, frontier, order = {start: None}, deque([start]), []
    expanded, generated, max_frontier = 0, 1, 1
    if start == goal:
        return _build("BFS", True, goal, parent, 0, 1, 1, order)
    while frontier:
        s = frontier.popleft()
        expanded += 1
        order.append(s)
        for to, action, cost in flow.edges.get(s, []):
            if to not in parent:
                parent[to] = (s, action, cost)
                generated += 1
                if to == goal:
                    return _build("BFS", True, goal, parent, expanded, generated, max_frontier, order)
                frontier.append(to)
                max_frontier = max(max_frontier, len(frontier))
    return _build("BFS", False, goal, parent, expanded, generated, max_frontier, order)


def dfs(flow, start, goal):
    """Depth-first graph search: always go on from the most recently found screen."""
    parent, explored, order = {}, set(), []
    stack = [(start, None)]
    expanded, generated, max_frontier = 0, 1, 1
    while stack:
        s, via = stack.pop()
        if s in explored:
            continue
        explored.add(s)
        parent[s] = via
        expanded += 1
        order.append(s)
        if s == goal:
            return _build("DFS", True, goal, parent, expanded, generated, max_frontier, order)
        for to, action, cost in reversed(flow.edges.get(s, [])):  # reversed so the first listed action is tried first
            if to not in explored:
                stack.append((to, (s, action, cost)))
                generated += 1
        max_frontier = max(max_frontier, len(stack))
    return _build("DFS", False, goal, parent, expanded, generated, max_frontier, order)


def ucs(flow, start, goal):
    """Uniform-cost search: always expand the screen reached with the least effort so far (goal test on expansion)."""
    parent, best, order = {start: None}, {start: 0.0}, []
    tie = itertools.count()
    heap = [(0.0, next(tie), start)]
    expanded, generated, max_frontier = 0, 1, 1
    done = set()
    while heap:
        g, _, s = heapq.heappop(heap)
        if s in done:
            continue
        done.add(s)
        expanded += 1
        order.append(s)
        if s == goal:
            return _build("UCS", True, goal, parent, expanded, generated, max_frontier, order)
        for to, action, cost in flow.edges.get(s, []):
            ng = g + cost
            if to not in best or ng < best[to]:
                best[to] = ng
                parent[to] = (s, action, cost)
                heapq.heappush(heap, (ng, next(tie), to))
                generated += 1
        max_frontier = max(max_frontier, len(heap))
    return _build("UCS", False, goal, parent, expanded, generated, max_frontier, order)


ALGORITHMS = {"BFS": bfs, "DFS": dfs, "UCS": ucs}


def compare(flow, start, goal):
    """Run all three searches: {'BFS': Plan, 'DFS': Plan, 'UCS': Plan}."""
    return {name: fn(flow, start, goal) for name, fn in ALGORITHMS.items()}


def goal_for_guide(flows, app_id, title):
    """(flow, goal_state) for a how-to guide that has a screen graph, else None."""
    flow = flows.get(app_id)
    if not flow:
        return None
    for g in flow.goals:
        if g["guide"] == title:
            return flow, g["state"]
    return None

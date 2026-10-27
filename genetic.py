"""A genetic algorithm (GA) for real-valued chromosomes.

  chromosome   a list of numbers, one gene per setting (here: the agent's thresholds and trust weights)
  fitness      a function that scores a chromosome (here: how well the agent does on example screens)
  selection    tournament: pick k random individuals, the fittest becomes a parent
  crossover    one-point (or uniform): the child takes genes from both parents
  mutation     each gene is nudged by Gaussian noise with a small probability, then kept inside its range
  elitism      the best few individuals are copied unchanged into the next generation
  seeding      known-good chromosomes (the default settings) can be put in the first population

It knows nothing about screens or windows, so it is tested separately on a toy problem."""
import random
from dataclasses import dataclass, field


@dataclass
class GAResult:
    best: list
    best_fitness: float
    history: list = field(default_factory=list)    # [{"generation", "best", "mean", "worst"}]
    evaluations: int = 0


def _clamp(v, ranges):
    return [min(max(x, lo), hi) for x, (lo, hi) in zip(v, ranges)]


def tournament(pop, fit, k, rng):
    """Index of the winner of a tournament of k randomly chosen individuals."""
    contenders = [rng.randrange(len(pop)) for _ in range(k)]
    return max(contenders, key=lambda i: fit[i])


def crossover(a, b, rng, kind="one_point"):
    if len(a) < 2:
        return list(a), list(b)
    if kind == "uniform":
        mask = [rng.random() < 0.5 for _ in a]
        return [x if m else y for x, y, m in zip(a, b, mask)], [y if m else x for x, y, m in zip(a, b, mask)]
    cut = rng.randint(1, len(a) - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def mutate(v, ranges, rate, scale, rng):
    """Each gene changes with probability `rate` by a Gaussian step of `scale` x its range."""
    out = list(v)
    for i, (lo, hi) in enumerate(ranges):
        if rng.random() < rate:
            out[i] += rng.gauss(0.0, scale * (hi - lo))
    return _clamp(out, ranges)


def evolve(fitness, ranges, pop_size=30, generations=20, crossover_rate=0.8, mutation_rate=0.15, mutation_scale=0.12,
           tournament_size=3, elite=2, seeds=(), rng=None, crossover_kind="one_point", on_generation=None):
    """Run the GA and return a GAResult. `ranges` is a list of (low, high), one per gene."""
    rng = rng or random.Random()
    cache = {}
    evals = [0]

    def score(v):
        key = tuple(round(x, 9) for x in v)
        if key not in cache:
            cache[key] = fitness(list(v))
            evals[0] += 1
        return cache[key]

    pop = [_clamp(list(s), ranges) for s in seeds][:pop_size]
    while len(pop) < pop_size:
        pop.append([rng.uniform(lo, hi) for lo, hi in ranges])

    history = []
    for gen in range(generations + 1):
        fit = [score(v) for v in pop]
        order = sorted(range(len(pop)), key=lambda i: -fit[i])
        history.append({"generation": gen, "best": fit[order[0]], "mean": sum(fit) / len(fit), "worst": fit[order[-1]]})
        if on_generation:
            on_generation(history[-1], pop[order[0]])
        if gen == generations:
            break
        nxt = [list(pop[i]) for i in order[:elite]]
        while len(nxt) < pop_size:
            a = pop[tournament(pop, fit, tournament_size, rng)]
            b = pop[tournament(pop, fit, tournament_size, rng)]
            if rng.random() < crossover_rate:
                c1, c2 = crossover(a, b, rng, crossover_kind)
            else:
                c1, c2 = list(a), list(b)
            nxt.append(mutate(c1, ranges, mutation_rate, mutation_scale, rng))
            if len(nxt) < pop_size:
                nxt.append(mutate(c2, ranges, mutation_rate, mutation_scale, rng))
        pop = nxt
    best = max(pop, key=score)
    return GAResult(list(best), score(best), history, evals[0])


def random_search(fitness, ranges, evaluations, rng=None, seeds=()):
    """Baseline: try the same number of random chromosomes and keep the best (to show the GA is doing more than luck)."""
    rng = rng or random.Random()
    best, best_f = None, float("-inf")
    candidates = [_clamp(list(s), ranges) for s in seeds]
    for i in range(evaluations):
        v = candidates[i] if i < len(candidates) else [rng.uniform(lo, hi) for lo, hi in ranges]
        f = fitness(v)
        if f > best_f:
            best, best_f = v, f
    return GAResult(best, best_f, [], evaluations)

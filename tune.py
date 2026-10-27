"""Tune the logical agent's settings with the genetic algorithm.     Run:  python tune.py

The agent has 12 settings (how close a word must be to the cursor, how similar a misread word must be,
how much each kind of clue is trusted, ...). Instead of guessing them, a genetic algorithm searches for good values:

  1. build made-up labelled screens (synth.py) and let the real sensors measure them once
  2. fitness of a chromosome = how well the agent does on those screens with those settings
  3. evolve a population of chromosomes, starting from the default settings plus random ones
  4. report accuracy on screens the GA never saw (a held-out set), then save the best settings

The tuned settings are only as good as the example screens: they are made-up, so treat the result as tuning for
this kind of screen, not a guarantee for every real one."""
import argparse
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import genetic
import params as pm
from knowledge import norm


def score_one(agent, p, sc):
    """Score one scenario: the app part and the item part are worth 0.5 each.
    right answer +0.5, 'I do not know' 0, a wrong answer -0.25 (a confident mistake is worse than admitting it)."""
    agent.params = p
    rec = agent.step(sc.raw, "ask")
    got = (rec.app["id"] if rec.app else None, norm(rec.element["names"][0]) if rec.element else None)
    total = 0.0
    for g, e in zip(got, sc.expect):
        if g == e:
            total += 0.5
        elif g is None:
            total += 0.0
        else:
            total -= 0.25
    return total


def exact(agent, p, sc):
    agent.params = p
    rec = agent.step(sc.raw, "ask")
    got = (rec.app["id"] if rec.app else None, norm(rec.element["names"][0]) if rec.element else None)
    return got == sc.expect


PULL = 0.1    # how strongly settings are pulled back towards the defaults (see fitness_on)


def distance_from_default(vector):
    """Mean squared distance from the default settings, each gene measured as a share of its range."""
    d = pm.to_vector(pm.Params())
    return sum(((x - y) / (pm.RANGES[g][1] - pm.RANGES[g][0])) ** 2 for x, y, g in zip(vector, d, pm.GENES)) / len(d)


def fitness_on(agent, scenarios, pull=PULL):
    """Average score on the example screens, minus a small penalty for straying from the defaults.
    The penalty is a prior: when two settings do equally well on the examples, prefer the ordinary-looking one,
    so a change is only made when the examples clearly support it (this fights over-fitting)."""
    def f(vector):
        p = pm.from_vector(vector)
        return sum(score_one(agent, p, sc) for sc in scenarios) / len(scenarios) - pull * distance_from_default(vector)
    return f


def accuracy(agent, p, scenarios):
    """Share of scenarios where both the app and the item are exactly right."""
    return sum(exact(agent, p, sc) for sc in scenarios) / max(len(scenarios), 1)


def by_kind(agent, p, scenarios):
    out = {}
    for sc in scenarios:
        a, b = out.get(sc.kind, (0, 0))
        out[sc.kind] = (a + int(exact(agent, p, sc)), b + 1)
    return out


def make_workbench(tmp):
    """A tutor that uses a scratch folder (so tuning never touches what you taught the real app) and has two taught icons."""
    from ai_tutor import Tutor
    from config import cfg
    import synth

    tutor = Tutor(cfg, learned_dir=Path(tmp), params=pm.Params())
    synth.teach_samples(tutor)
    return tutor


def run_tuning(n_scenarios=120, pop=30, generations=20, seed=7, log=print, tutor=None):
    import synth

    tmp = tempfile.mkdtemp()
    tutor = tutor or make_workbench(tmp)
    t0 = time.time()
    scenarios = synth.build_scenarios(tutor, n_scenarios, seed=seed)
    log(f"Measured {len(scenarios)} made-up screens in {time.time() - t0:.0f}s")
    rng = random.Random(seed)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.7)
    train, test = shuffled[:cut], shuffled[cut:]
    agent = tutor.agent
    default = pm.Params()

    fit = fitness_on(agent, train)
    ranges = [pm.RANGES[g] for g in pm.GENES]
    t1 = time.time()

    def show(h, _best):
        log(f"  generation {h['generation']:>2}: best {h['best']:.3f}   average {h['mean']:.3f}")

    result = genetic.evolve(fit, ranges, pop_size=pop, generations=generations, seeds=[pm.to_vector(default)],
                            rng=random.Random(seed), on_generation=show)
    baseline = genetic.random_search(fit, ranges, result.evaluations, rng=random.Random(seed), seeds=[pm.to_vector(default)])
    tuned = pm.from_vector(result.best)
    info = {
        "scenarios": len(scenarios), "train": len(train), "held_out": len(test), "seed": seed,
        "population": pop, "generations": generations, "evaluations": result.evaluations,
        "seconds": round(time.time() - t1, 1),
        "fitness_default_train": round(fit(pm.to_vector(default)), 4), "fitness_tuned_train": round(result.best_fitness, 4),
        "fitness_random_search_train": round(baseline.best_fitness, 4), "pull_towards_default": PULL,
        "accuracy_default_train": round(accuracy(agent, default, train), 4),
        "accuracy_tuned_train": round(accuracy(agent, tuned, train), 4),
        "accuracy_default_held_out": round(accuracy(agent, default, test), 4),
        "accuracy_tuned_held_out": round(accuracy(agent, tuned, test), 4),
        "history": result.history,
        "by_kind_default_held_out": by_kind(agent, default, test),
        "by_kind_tuned_held_out": by_kind(agent, tuned, test),
    }
    return tuned, info, tutor


def main():
    ap = argparse.ArgumentParser(description="Tune the logical agent's settings with a genetic algorithm")
    ap.add_argument("--scenarios", type=int, default=120)
    ap.add_argument("--population", type=int, default=30)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--force", action="store_true", help="save the tuned settings even if they are not better on unseen screens")
    args = ap.parse_args()

    from config import LEARNED_DIR
    from ai_tutor import TUNED_FILE

    tuned, info, _ = run_tuning(args.scenarios, args.population, args.generations, args.seed)
    print()
    print(f"Exact-answer accuracy on screens used for tuning : default {info['accuracy_default_train']:.0%}  ->  tuned {info['accuracy_tuned_train']:.0%}")
    print(f"Exact-answer accuracy on screens never seen      : default {info['accuracy_default_held_out']:.0%}  ->  tuned {info['accuracy_tuned_held_out']:.0%}")
    print(f"(fitness {info['fitness_default_train']:.3f} -> {info['fitness_tuned_train']:.3f}; random search with the same effort reached {info['fitness_random_search_train']:.3f})")
    print("Tuned settings:")
    for k, v in asdict(tuned).items():
        print(f"  {k:12s} {v:.3f}   (default {getattr(pm.Params(), k):.3f})")
    better = info["accuracy_tuned_held_out"] > info["accuracy_default_held_out"]
    if args.no_save:
        return
    if better or args.force:
        pm.save(LEARNED_DIR / TUNED_FILE, tuned, info)
        print(f"\nSaved to {LEARNED_DIR / TUNED_FILE}. The app uses these settings the next time it starts.")
    else:
        print("\nThe tuned settings were NOT better on screens the GA never saw, so the default settings are kept.\n"
              "(Use --force to save them anyway, or try more screens: --scenarios 300.)")


if __name__ == "__main__":
    main()

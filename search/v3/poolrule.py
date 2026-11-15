# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""poolrule.py -- the elite pool's pure rules (#188, RULES-pool.md): the join decision, the worst member, the re-seed
ranking, the reselection and the parent choice. No I/O: the driver (driver.py) builds the statistics (seqrule.Paired)
and calls these; test_pool188.py checks them. With K = 1 the driver does not call them at all (today's rule).

Notation: tags are the driver's candidate tags; `da[t]` is a member's size-adjusted paired statistic against the
incumbent (d + off, the driver's Paired.da(); the incumbent is 0); `acc[t]` is the driver's accept rule vs the
incumbent (Paired.final() == "accept": d + off < 0 on every part and the (c) guard)."""

def join_action(full, acc_inc, acc_worst):
    """RULES-pool §3 for a whole-slice candidate. Returns (mode, inc): mode "add" (the pool is not full and the
    candidate beats the incumbent: it joins and becomes incumbent, the old one stays), "replace" (the pool is full and
    the candidate beats the worst or the incumbent: it replaces the worst; it becomes incumbent if it beats the
    incumbent), or None (reject)."""
    if not full:
        return ("add", True) if acc_inc else (None, False)
    if acc_worst or acc_inc:
        return ("replace", bool(acc_inc))
    return (None, False)


def worst_of(inc, da):
    """the member with the largest size-adjusted statistic against the incumbent (RULES-pool §2); a tie goes to the
    non-incumbent member. `da` maps each member's tag to its statistic (the incumbent's is 0)."""
    return max(da, key=lambda t: (da[t], t != inc))


def reseed_pick(inc, scored, K):
    """RULES-pool §6: scored = [(tag, da, acc)] for the re-seed candidates other than the incumbent (against it, on the
    slice). The ranking puts the incumbent (0) among them, ties to the incumbent. The new incumbent is the first-ranked
    candidate that passes the accept rule (the old incumbent if none does). The pool is the top K of the ranking; the
    new incumbent is kept in it (replacing the last member if it is outside the top K). Returns (new_inc, pool)."""
    ranked = sorted([(0.0, inc, True)] + [(float(da), t, bool(a)) for t, da, a in scored],
                    key=lambda x: (x[0], x[1] != inc))
    new = next((t for _, t, a in ranked if a and t != inc), inc)
    pool = [t for _, t, _ in ranked][:K]
    if new not in pool: pool = pool[:max(K - 1, 0)] + [new]
    return new, pool


def reselect_pick(inc, scored):
    """RULES-pool §5 (rotation re-selection): scored = [(tag, da, acc)] for the members other than the incumbent, on the
    new slice. The accepted member with the lowest statistic becomes incumbent; None if no member is accepted."""
    ok = [(da, t) for t, da, a in scored if a and t != inc]
    return min(ok, key=lambda x: (x[0], x[1]))[1] if ok else None


def choose_parent(inc, members, rng, p_inc=0.6):
    """RULES-pool §4: the incumbent with probability p_inc, else uniform over the other members (draws only when the pool
    has more than one member)"""
    others = [t for t in members if t != inc]
    if not others: return inc
    if rng.random() < p_inc: return inc
    return others[int(rng.random() * len(others)) % len(others)]


def choose_other(parent, members, rng):
    """the second parent of a cross child: uniform over the members other than `parent` (None if there is none)"""
    others = [t for t in members if t != parent]
    if not others: return None
    return others[int(rng.random() * len(others)) % len(others)]


def members_from_events(events):
    """the pool's membership as the driver's replay rebuilds it (test helper): last 'pool' event per size wins"""
    out = {}
    for e in events:
        if e.get("ev") == "pool": out[e["size"]] = list(e["members"])
    return out


__all__ = ["join_action", "worst_of", "reseed_pick", "reselect_pick", "choose_parent", "choose_other", "members_from_events"]

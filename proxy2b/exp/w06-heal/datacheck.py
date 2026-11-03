# datacheck.py -- appended to heal.py's functions (its main guard removed) and run CPU-only on the 4070's data:
# the training windows, the held-out slice and the oracle set heal.py would use (counts, scored share, pools, overlap).
import collections
meta = Meta(CAL); pools = json.load(open(POOLS))["pool"]; assert len(pools) == len(meta.docs)
gold = Meta(GOLD); gorig = {d["orig"] for d in gold.docs}
it = qad_items(meta, pools, 1024, 2048, 0, 8); held = heldout_items(meta, pools, GOLD, 48, 2048, 1); orc = golden_items(gold)
def stat(m, items):
    nt = sum(n for _, _, n in items); ns = sum(len(m.window(*x)[1]) for x in items)
    return nt, ns
nt, ns = stat(meta, it); ht, hs = stat(meta, held); ot, os_ = stat(gold, orc)
src = collections.Counter(meta.docs[i]["source"].split("/")[0] for i, _, _ in it)
print(f"train: {len(it)} windows from {len({i for i, _, _ in it})} qad docs, {nt} tokens, {ns} scored rows ({ns / nt:.1%}); "
      f"pools {set(pools[i] for i, _, _ in it)}; windows starting mid-doc {sum(a > 0 for _, a, _ in it)}")
print(f"  top sources: {src.most_common(8)}")
print(f"held-out: {len(held)} search docs, {ht} tokens, {hs} scored rows; pools {set(pools[i] for i, _, _ in held)}; "
      f"overlap with the golden {len({i for i, _, _ in held} & gorig)}")
print(f"oracle: {len(orc)} golden docs, {ot} tokens, {os_} scored rows (golden gen rows {int(gold.gen.sum())})")
q = sum(1 for i, d in enumerate(meta.docs) if pools[i] == "qad")
print(f"qad docs {q}; available windows at per_doc 8: {len(qad_items(meta, pools, 10**9, 2048, 0, 8))}")

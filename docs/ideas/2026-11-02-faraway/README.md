# Faraway lens (2026-11-02): generated, untested

One Opus 5.5 (xhigh) agent asked for approaches that would **replace large parts** of our pipeline rather than tune
it, to avoid a local minimum. Same context as the [2026-11-02 pass](../2026-11-02/) (plan docs, experiment index and
log, 448-paper corpus) plus that pass's ranked list, which it was told not to repeat. No critic: read directly.

Unlike the other lenses it ran its own CPU measurements on the 27B weights (§0 of `faraway.md`): the trunk is close
to its floor where most bytes are (gate/up within ~0.1 bit of an idealized LDLQ + Gaussian coder), the weights are
near-Gaussian, there is no cross-layer redundancy, but within a layer a down_proj column is partly predictable from
its neuron's up / gate rows (0.09–0.23 bit). These measurements are the agent's own, not re-checked.

**Constraint added afterwards:** the challenge targets VRAM size, so storage-only gains need kernels that run from
the stored form (see the note at the end of the report). Status of anything tried: [`../../experiments/`](../../experiments/).

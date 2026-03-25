## Layer 3 is not a general refinement stage. It is a sparse correction-event detector.

Some gate-open cases are already sitting on a stable answer, so a tiny extra correction after the gate opens is usually just noise.

The non-obvious part is that the dangerous cases are not the ones asking for a big move.

In this experiment, the bigger correction moves are often the helpful ones, because they signal that the anchored answer is still missing a real piece of the pattern.

The holdout miss happens when the solver refines by default even though the proposed move is too small to carry real signal.

That means the next Layer 3 should not refine whenever refinement is allowed.

It should refine only when the correction itself is strong enough to be real, and otherwise stay anchored.

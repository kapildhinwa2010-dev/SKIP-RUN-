# SKIP-RUN

[![Stars](https://img.shields.io/github/stars/kapildhinwa2010-dev/SKIP-RUN-?style=flat-square)](https://github.com/kapildhinwa2010-dev/SKIP-RUN-/stargazers) [![Forks](https://img.shields.io/github/forks/kapildhinwa2010-dev/SKIP-RUN-?style=flat-square)](https://github.com/kapildhinwa2010-dev/SKIP-RUN-/network/members) [![Issues](https://img.shields.io/github/issues/kapildhinwa2010-dev/SKIP-RUN-?style=flat-square)](https://github.com/kapildhinwa2010-dev/SKIP-RUN-/issues) [![Top language](https://img.shields.io/github/languages/top/kapildhinwa2010-dev/SKIP-RUN-?style=flat-square)](https://github.com/kapildhinwa2010-dev/SKIP-RUN-)

Depth fakeness test: skip every k-th block of a stack and measure how much the behavior actually changes.

Deep stacks often contain layers that do almost nothing. SKIP-RUN runs a stack of additive blocks complete, then again with every k-th block removed. Small deltas mean that part of the depth is just numbers piled up, not real computation.

## What it does

- Runs a stack of additive blocks (x = x + block(x)) end to end.
- Skips every k-th block for a sweep of k values.
- Reports the removed indices, the score delta, and the retained fraction.
- The real_depth helper returns the largest k that stays within a tolerance.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from skip_run.depth import skip_sweep, real_depth

blocks = [make_block(i) for i in range(12)]
report = skip_sweep(blocks, np.zeros(16), score_fn=lambda v: float(v[0]))
print(real_depth(report, tolerance=0.05))
```

## How it works

When every k-th block is removed, the surviving stack is run again from the same input. If the score barely moves, those blocks were not contributing. A stack where k = 2 is safe but k = 3 is not has a very different depth profile than one where even k = 1 barely matters.

## Honest scope

- Blocks must be additive; the tool never calls the blocks on shared state.
- The verdict depends entirely on the score function you pass in.
- Skipping is positional (every k-th), not a search for the best subset.

## License

MIT
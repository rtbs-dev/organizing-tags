from DEPRECATED.recovery_experiment import ex
import numpy as np
from itertools import product

import warnings
warnings.warn("the grid_search module is deprecated!", DeprecationWarning,
              stacklevel=2)

def gen_log_space(limit, n):
    """from SO user Avaris (https://stackoverflow.com/a/12421820)"""

    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result) < n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value
            # by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values
            # will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)


if __name__ == "__main__":

    nwos_ch = gen_log_space(5001, 16)[4:]
    nitems_ch = [2, 3, 4, 5]
    model_ch = ['drivetrain', 'aircraft', 'bicycle']

    for nwos, nitems, model in product(nwos_ch, nitems_ch, model_ch):
        print(nwos, nitems, model)
        ex.run(config_updates={
            'model': model,
            'nwos': nwos,
            'nitems': nitems,
        })

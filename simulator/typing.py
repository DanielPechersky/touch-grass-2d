from typing import Literal

import numpy as np

type ChainIdx = int
type ChainSize = int
type Chains = np.ndarray[tuple[ChainIdx, ChainSize, Literal[2]], np.dtypes.Float32DType]

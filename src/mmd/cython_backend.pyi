import numpy as np
from typing import Any

def sig_kernel_batch_varpar(
    G_static: Any, _naive_solver: bool = False, pad: bool = False
) -> np.ndarray: ...
def sig_kernel_Gram_varpar(
    G_static: Any,
    sym: bool = False,
    _naive_solver: bool = False,
    pad_dim1: bool = False,
    pad_dim2: bool = False,
) -> np.ndarray: ...

# _shared.py — module-level settings, path setup and shared imports for this package (was _module.py).
import copy
from typing import Callable, Optional, Union
import mlx.core as mx
from _convert_io import EVAL_EVERY, materialize

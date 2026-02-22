"""Shared type aliases for KQuity codebase."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Coord: TypeAlias = tuple[int, int]

GameStateVector: TypeAlias = npt.NDArray[np.float32]
GameStatesMatrix: TypeAlias = npt.NDArray[np.float32]
OutcomesLabelVector: TypeAlias = npt.NDArray[np.int8]
GameIdArray: TypeAlias = npt.NDArray[np.int64]
TimestampArray: TypeAlias = npt.NDArray[np.float32]
MaterializeResult: TypeAlias = tuple[
    GameStatesMatrix, OutcomesLabelVector, GameIdArray, TimestampArray
]

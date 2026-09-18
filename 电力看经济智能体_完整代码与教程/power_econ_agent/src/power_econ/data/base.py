from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError

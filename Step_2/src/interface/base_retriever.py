from abc import ABC, abstractmethod
from typing import List

from .base_datastore import BaseDatastore


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[str]:
        pass

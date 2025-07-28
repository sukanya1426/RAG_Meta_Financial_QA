from abc import ABC, abstractmethod
from typing import List

from .base_datastore import DataItem  # Changed from interface.base_datastore to .base_datastore


class BaseIndexer(ABC):

    @abstractmethod
    def index(self, document_paths: List[str]) -> List[DataItem]:
        pass

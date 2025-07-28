from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field


class DataItem(BaseModel):
    content: str = ""
    source: str = ""
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the content")


class BaseDatastore(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset/Initialize the datastore."""
        pass

    @abstractmethod
    def add_items(self, items: List[DataItem]) -> None:
        """Add items to the datastore."""
        pass

    @abstractmethod
    def get_vector(self, content: str) -> List[float]:
        """Generate vector embedding for the given content."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for similar content using vector similarity."""
        pass

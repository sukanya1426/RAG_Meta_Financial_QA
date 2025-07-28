from ..interface.base_datastore import BaseDatastore  # Changed
from ..interface.base_retriever import BaseRetriever  # Changed

# Rest of the code remains the same

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return self.datastore.search(query, top_k=top_k)
from typing import List
from ..interface.base_datastore import BaseDatastore, DataItem
import lancedb
from lancedb.table import Table
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Datastore(BaseDatastore):
    DB_PATH = "data/sample-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        self.vector_dimensions = 384  # Dimension for all-MiniLM-L6-v2
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = self.embedding_model.tokenizer  # Access tokenizer
        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table: Table = self._get_table()

    def reset(self) -> Table:
        try:
            # Fully remove the table directory to clear any corrupted files
            table_dir = os.path.join(self.DB_PATH, f"{self.DB_TABLE_NAME}.lance")
            if os.path.exists(table_dir):
                shutil.rmtree(table_dir)
                logger.info(f"Removed existing table directory: {table_dir}")
            else:
                logger.info(f"No existing table directory found: {table_dir}")
        except Exception as e:
            logger.error(f"Error removing table directory: {str(e)}")
            raise
        
        # Ensure the parent directory exists and is writable
        os.makedirs(self.DB_PATH, exist_ok=True)
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.vector_dimensions)),
            pa.field("content", pa.utf8()),
            pa.field("source", pa.utf8()),
        ])
        self.vector_db.create_table(self.DB_TABLE_NAME, schema=schema)
        logger.info(f"✅ Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
        return self.vector_db.open_table(self.DB_TABLE_NAME)

    def get_vector(self, content: str) -> List[float]:
        try:
            # Truncate to 512 tokens to avoid exceeding model limit
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            if len(tokens) > 512:
                logger.warning(f"Content exceeds 512 tokens, truncating to 512 tokens")
                tokens = tokens[:512]
                content = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            embedding = self.embedding_model.encode(content).tolist()
            if len(embedding) != self.vector_dimensions:
                logger.error(f"Invalid embedding dimension: {len(embedding)} != {self.vector_dimensions}")
                raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.vector_dimensions}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for content: {str(e)}")
            raise

    def add_items(self, items: List[DataItem]) -> None:
        if not items:
            logger.warning("No items provided to add to datastore")
            return
        
        logger.info(f"Adding {len(items)} items to datastore")
        with ThreadPoolExecutor(max_workers=8) as executor:
            entries = []
            for i, item in enumerate(items):
                try:
                    if not item.content or not isinstance(item.content, str):
                        logger.warning(f"Skipping item {i}: Invalid or empty content")
                        continue
                    if item.embedding is None:
                        item.embedding = self.get_vector(item.content)
                    if len(item.embedding) != self.vector_dimensions:
                        logger.error(f"Invalid embedding dimension for item {i}: {len(item.embedding)}")
                        continue
                    entries.append({
                        "vector": item.embedding,
                        "content": item.content,
                        "source": item.source,
                    })
                    logger.debug(f"Prepared item {i}: {item.content[:50]}...")
                except Exception as e:
                    logger.error(f"Error preparing item {i}: {str(e)}")
                    continue
        
        if not entries:
            logger.error("No valid entries to insert into datastore")
            return

        try:
            if not self.table:
                logger.warning("Table is invalid, resetting datastore")
                self.table = self.reset()
            
            self.table.merge_insert("source")\
                     .when_matched_update_all()\
                     .when_not_matched_insert_all()\
                     .execute(entries)
            logger.info(f"✅ Added {len(entries)} items to the datastore")
        except Exception as e:
            logger.error(f"Error inserting items into datastore: {str(e)}")
            if "Not found" in str(e) or "IO" in str(e):
                logger.info("Attempting to reset datastore due to I/O error")
                self.table = self.reset()
                try:
                    self.table.merge_insert("source")\
                             .when_matched_update_all()\
                             .when_not_matched_insert_all()\
                             .execute(entries)
                    logger.info(f"✅ Successfully added {len(entries)} items after reset")
                except Exception as e2:
                    logger.error(f"Retry failed: {str(e2)}")
                    raise
            else:
                raise

    def search(self, query: str, top_k: int = 5) -> List[str]:
        try:
            vector = self.get_vector(query)
            logger.debug(f"Query vector length: {len(vector)}")
            results = (
                self.table.search(vector)
                .metric("cosine")  # Use cosine similarity
                .select(["content", "source"])
                .limit(top_k)
                .to_list()
            )
            if not results and self.table:
                logger.warning("No results found, attempting table reset")
                self.table = self.reset()
                results = (
                    self.table.search(vector)
                    .metric("cosine")
                    .select(["content", "source"])
                    .limit(top_k)
                    .to_list()
                )
            logger.info(f"Search returned {len(results)} results for query: {query[:50]}...")
            logger.debug(f"Top result content: {results[0].get('content')[:100] if results else 'None'}")
            return [result.get("content") for result in results]
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            if "Not found" in str(e) or "IO" in str(e):
                logger.info("I/O error during search, resetting datastore")
                self.table = self.reset()
                try:
                    vector = self.get_vector(query)
                    results = (
                        self.table.search(vector)
                        .metric("cosine")
                        .select(["content", "source"])
                        .limit(top_k)
                        .to_list()
                    )
                    logger.info(f"Search after reset returned {len(results)} results")
                    logger.debug(f"Top result content: {results[0].get('content')[:100] if results else 'None'}")
                    return [result.get("content") for result in results]
                except Exception as e2:
                    logger.error(f"Retry failed: {str(e2)}")
                    return []
            return []

    def _get_table(self) -> Table:
        try:
            table = self.vector_db.open_table(self.DB_TABLE_NAME)
            logger.info(f"Opened existing table: {self.DB_TABLE_NAME}")
            return table
        except Exception as e:
            logger.info(f"Table not found, resetting datastore: {e}")
            return self.reset()
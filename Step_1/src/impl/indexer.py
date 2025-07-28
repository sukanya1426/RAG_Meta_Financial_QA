import os
from typing import List
from ..interface.base_datastore import DataItem
from ..interface.base_indexer import BaseIndexer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, DocChunk
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Indexer(BaseIndexer):
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(chunk_size=400, chunk_overlap=50)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _token_based_chunking(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into chunks based on token count using the model's tokenizer."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text.strip())
        return chunks

    def index(self, document_paths: List[str]) -> List[DataItem]:
        items = []
        for document_path in document_paths:
            logger.info(f"Processing document: {document_path}")
            if not os.path.exists(document_path):
                logger.error(f"Document not found: {document_path}")
                continue

            try:
                logger.info("Converting document...")
                convert_result = self.converter.convert(document_path)
                document = convert_result.document
                logger.info(f"Document converted successfully. Pages: {document.num_pages}")

                # Extract text with multiple fallbacks
                text = document.export_to_text()
                if not text or len(text.strip()) < 10:
                    logger.warning("No text extracted via export_to_text, attempting fallbacks...")
                    text_parts = []
                    # Fallback 1: Body items
                    if hasattr(document, 'body') and document.body:
                        for item in document.body:
                            if hasattr(item, 'text') and item.text:
                                text_parts.append(item.text.strip())
                    # Fallback 2: Page-level text
                    if hasattr(document, 'pages') and document.pages:
                        for page in document.pages:
                            if hasattr(page, 'text') and page.text:
                                text_parts.append(page.text.strip())
                            elif hasattr(page, 'content') and page.content:
                                for content in page.content:
                                    if hasattr(content, 'text') and content.text:
                                        text_parts.append(content.text.strip())
                    # Fallback 3: OCR content
                    if hasattr(document, 'pages') and document.pages:
                        for page in document.pages:
                            if hasattr(page, 'ocr') and page.ocr:
                                text_parts.append(page.ocr.strip())
                    text = '\n\n'.join([part for part in text_parts if part]).strip()
                
                logger.info(f"Extracted text length: {len(text)}")
                logger.debug(f"Extracted text preview: {text[:200]}...")
                if not text:
                    logger.error("Failed to extract any text from document")
                    continue

                logger.info("Chunking document...")
                # Try HybridChunker first
                chunks = list(self.chunker.chunk(document))  # Convert to list to avoid iterator
                logger.debug(f"Type of chunks: {type(chunks)}")
                logger.debug(f"Number of chunks from HybridChunker: {len(chunks)}")

                if not chunks:
                    logger.warning("HybridChunker produced no chunks, using token-based chunking...")
                    # Fall back to token-based chunking
                    chunk_texts = self._token_based_chunking(text)
                    chunks = [DocChunk(text=chunk_text, meta=document.meta) for chunk_text in chunk_texts]
                
                logger.info(f"Generated {len(chunks)} chunks")
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Chunk {i}: {chunk.text[:100]}...")

                if not chunks:
                    logger.error("No chunks generated from document")
                    continue

                new_items = self._items_from_chunks(chunks, document_path)
                logger.info(f"Created {len(new_items)} items")
                items.extend(new_items)

            except Exception as e:
                logger.error(f"Error processing document {document_path}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Total items created: {len(items)}")
        return items

    def _items_from_chunks(self, chunks: List[DocChunk], document_path: str) -> List[DataItem]:
        items = []
        for i, chunk in enumerate(chunks):
            try:
                text = chunk.text.strip()
                if not text:
                    logger.warning(f"Empty chunk {i} skipped")
                    continue

                source = f"{document_path}:chunk_{i}"
                item = DataItem(content=text, source=source)
                items.append(item)
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} chunks")

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        return items
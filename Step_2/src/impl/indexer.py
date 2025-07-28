import os
import pandas as pd
import json
from typing import List, Dict, Any
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

    def _extract_tables(self, document) -> List[Dict[str, Any]]:
        """Extract tables from the document and convert to structured format."""
        tables = []
        try:
            if hasattr(document, 'tables') and document.tables:
                logger.info(f"Found {len(document.tables)} tables in document")
                for i, table in enumerate(document.tables):
                    try:
                        # Extract table data
                        table_data = self._table_to_structured_data(table)
                        if table_data:
                            tables.append({
                                'table_id': f'table_{i}',
                                'data': table_data,
                                'text_representation': self._table_to_text(table_data),
                                'type': 'financial_table'
                            })
                            logger.info(f"Extracted table {i} with {len(table_data)} rows")
                    except Exception as e:
                        logger.error(f"Error processing table {i}: {str(e)}")
                        continue
            else:
                logger.info("No tables found in document")
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
        
        return tables

    def _table_to_structured_data(self, table) -> List[Dict[str, Any]]:
        """Convert table to structured data format."""
        try:
            rows = []
            headers = []
            
            # Extract headers and data from table cells
            if hasattr(table, 'table_cells') and table.table_cells:
                # Group cells by rows
                row_data = {}
                for cell in table.table_cells:
                    row_idx = getattr(cell, 'start_row_offset_idx', 0)
                    col_idx = getattr(cell, 'start_col_offset_idx', 0)
                    text = getattr(cell, 'text', '').strip()
                    
                    if row_idx not in row_data:
                        row_data[row_idx] = {}
                    row_data[row_idx][col_idx] = text
                
                # Convert to structured format
                if row_data:
                    # First row as headers
                    if 0 in row_data:
                        headers = [row_data[0].get(i, f'col_{i}') for i in range(max(row_data[0].keys()) + 1)]
                    
                    # Data rows
                    for row_idx in sorted(row_data.keys())[1:]:  # Skip header row
                        row = {}
                        for col_idx, header in enumerate(headers):
                            value = row_data[row_idx].get(col_idx, '')
                            # Try to parse numerical values
                            parsed_value = self._parse_financial_value(value)
                            row[header] = parsed_value
                        rows.append(row)
            
            return rows
        except Exception as e:
            logger.error(f"Error converting table to structured data: {str(e)}")
            return []

    def _parse_financial_value(self, value: str) -> Any:
        """Parse financial values (e.g., '$12,345', '27%') to appropriate types."""
        if not value or not isinstance(value, str):
            return value
        
        value = value.strip()
        
        # Handle percentage
        if value.endswith('%'):
            try:
                return float(value[:-1])
            except ValueError:
                return value
        
        # Handle currency (remove $ and commas)
        if value.startswith('$') or ',' in value:
            cleaned = value.replace('$', '').replace(',', '').strip()
            try:
                return float(cleaned)
            except ValueError:
                return value
        
        # Handle parentheses (negative numbers)
        if value.startswith('(') and value.endswith(')'):
            try:
                return -float(value[1:-1].replace(',', ''))
            except ValueError:
                return value
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def _table_to_text(self, table_data: List[Dict[str, Any]]) -> str:
        """Convert structured table data to text representation."""
        if not table_data:
            return ""
        
        text_parts = []
        headers = list(table_data[0].keys()) if table_data else []
        
        # Add headers
        text_parts.append(" | ".join(headers))
        text_parts.append("-" * 50)
        
        # Add data rows
        for row in table_data:
            row_text = " | ".join(str(row.get(header, '')) for header in headers)
            text_parts.append(row_text)
        
        return "\n".join(text_parts)

    def _token_based_chunking(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into chunks based on token count using the model's tokenizer."""
        # Note: We need to access tokenizer from the model
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
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

                # Extract structured data (tables)
                tables = self._extract_tables(document)
                logger.info(f"Extracted {len(tables)} tables")

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
                    text = '\n\n'.join([part for part in text_parts if part]).strip()
                
                logger.info(f"Extracted text length: {len(text)}")
                
                if not text and not tables:
                    logger.error("Failed to extract any text or tables from document")
                    continue

                # Process text chunks
                text_items = []
                if text:
                    logger.info("Chunking document text...")
                    chunks = list(self.chunker.chunk(document))
                    
                    if not chunks:
                        logger.warning("HybridChunker produced no chunks, using token-based chunking...")
                        chunk_texts = self._token_based_chunking(text)
                        chunks = [DocChunk(text=chunk_text, meta=document.meta) for chunk_text in chunk_texts]
                    
                    logger.info(f"Generated {len(chunks)} text chunks")
                    text_items = self._items_from_chunks(chunks, document_path, 'text')

                # Process table items
                table_items = []
                for table in tables:
                    table_item = DataItem(
                        content=table['text_representation'],
                        source=f"{document_path}:{table['table_id']}",
                        metadata={
                            'type': 'table',
                            'structured_data': table['data'],
                            'table_id': table['table_id']
                        }
                    )
                    table_items.append(table_item)
                
                logger.info(f"Created {len(text_items)} text items and {len(table_items)} table items")
                items.extend(text_items)
                items.extend(table_items)

            except Exception as e:
                logger.error(f"Error processing document {document_path}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Total items created: {len(items)}")
        return items

    def _items_from_chunks(self, chunks: List[DocChunk], document_path: str, content_type: str = 'text') -> List[DataItem]:
        items = []
        for i, chunk in enumerate(chunks):
            try:
                text = chunk.text.strip()
                if not text:
                    logger.warning(f"Empty chunk {i} skipped")
                    continue

                source = f"{document_path}:chunk_{i}"
                item = DataItem(
                    content=text, 
                    source=source,
                    metadata={
                        'type': content_type,
                        'chunk_index': i
                    }
                )
                items.append(item)
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} chunks")

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        return items
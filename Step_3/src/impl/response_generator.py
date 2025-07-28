from typing import List, Dict, Any
from ..interface.base_response_generator import BaseResponseGenerator
from ..util.invoke_ai import invoke_ai
import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a financial analyst assistant that provides accurate information based on Meta's financial reports.
Your task is to answer questions precisely using both text context and structured data.

Rules:
1. Only answer based on the given context and structured data
2. Use exact numbers and percentages when present
3. If comparing data (e.g., Q1 2024 vs Q1 2023), provide specific numbers and percentage changes
4. Format financial numbers consistently (e.g., "$X billion" or "$X million")
5. When structured data is available, prioritize it for numerical accuracy
6. If you're unsure, say "I cannot find this information in the provided context"
7. Keep responses concise and focused
"""

HYBRID_PROMPT_TEMPLATE = """
Text context: {text_context}

Structured data: {structured_data}

Financial data: {financial_data}

Answer the query: {query}

Based on the above information, provide a comprehensive and accurate response.
"""

class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using the Gemini API with text context only."""
        formatted_context = "\n\nRelevant sections:\n" + "\n---\n".join(context)
        user_message = f"Question: {query}\n\nContext:{formatted_context}\n\nAnswer based on the above context only:"
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)
    
    def generate_hybrid_response(self, query: str, search_results: Dict[str, Any]) -> str:
        """Generate a response using both text context and structured data."""
        logger.info("Generating hybrid response with structured data")
        
        # Extract components - handle both old and new formats
        text_context = search_results.get('text_context', [])
        structured_data = search_results.get('structured_data', [])
        financial_data = search_results.get('financial_data', [])
        
        # If we have search_results (new format), extract content by type
        if 'search_results' in search_results and search_results['search_results']:
            candidates = search_results['search_results']
            text_context = []
            structured_data = []
            financial_data = []
            
            for candidate in candidates:
                content = candidate.get('content', '')
                content_type = candidate.get('content_type', 'text')
                
                if content_type == 'text':
                    text_context.append(content)
                elif content_type == 'table':
                    structured_data.append(candidate)
                elif content_type in ['financial', 'financial_text']:
                    financial_data.append(candidate)
                else:
                    # Default to text if unknown type
                    text_context.append(content)
        
        # Format text context
        text_context_str = "\n---\n".join(text_context) if text_context else "No relevant text context found."
        
        # Format structured data
        structured_data_str = "No structured data found."
        if structured_data:
            structured_parts = []
            for data in structured_data:
                if 'data' in data:
                    table_str = self._format_table_data(data['data'])
                    structured_parts.append(f"Table from {data.get('source', 'unknown')}:\n{table_str}")
            structured_data_str = "\n\n".join(structured_parts) if structured_parts else "No structured data found."
        
        # Format financial data
        financial_data_str = "No specific financial data found."
        if financial_data:
            financial_parts = []
            for data in financial_data:
                content = data.get('content', '')
                if data.get('type') == 'financial_table' and 'structured_data' in data:
                    table_str = self._format_table_data(data['structured_data'])
                    financial_parts.append(f"Financial table:\n{table_str}")
                elif content:
                    financial_parts.append(content)
            financial_data_str = "\n\n".join(financial_parts) if financial_parts else "No specific financial data found."
        
        # Create the hybrid prompt
        user_message = HYBRID_PROMPT_TEMPLATE.format(
            text_context=text_context_str,
            structured_data=structured_data_str,
            financial_data=financial_data_str,
            query=query
        )
        
        logger.debug(f"Hybrid prompt length: {len(user_message)}")
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)
    
    def _format_table_data(self, table_data: List[Dict[str, Any]]) -> str:
        """Format structured table data for better readability."""
        if not table_data:
            return "Empty table"
        
        try:
            # Get headers
            headers = list(table_data[0].keys()) if table_data else []
            
            # Create table string
            lines = []
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
            
            for row in table_data:
                row_values = []
                for header in headers:
                    value = row.get(header, '')
                    # Format numbers nicely
                    if isinstance(value, (int, float)):
                        if abs(value) >= 1000:
                            row_values.append(f"{value:,.0f}")
                        else:
                            row_values.append(str(value))
                    else:
                        row_values.append(str(value))
                lines.append(" | ".join(row_values))
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error formatting table data: {str(e)}")
            return str(table_data)
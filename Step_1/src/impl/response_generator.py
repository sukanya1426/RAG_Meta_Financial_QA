from typing import List
from ..interface.base_response_generator import BaseResponseGenerator
from ..util.invoke_ai import invoke_ai  # Keep only this one import

SYSTEM_PROMPT = """
You are a financial analyst assistant that provides accurate information based on Meta's financial reports.
Your task is to answer questions precisely using only the provided context.

Rules:
1. Only answer based on the given context
2. Use exact numbers and percentages when present
3. If you're unsure, say "I cannot find this information in the provided context"
4. Format financial numbers consistently (e.g., "$X billion")
5. Keep responses concise and focused
"""

class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using the Gemini API."""
        formatted_context = "\n\nRelevant sections:\n" + "\n---\n".join(context)
        user_message = f"Question: {query}\n\nContext:{formatted_context}\n\nAnswer based on the above context only:"
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)
from typing import List
from interface.base_response_generator import BaseResponseGenerator
from invoke_ai import invoke_ai

SYSTEM_PROMPT = """
You are an assistant that answers questions based on provided context.
"""

class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using the Gemini API."""
        context_text = "\n".join(context)
        user_message = f"Based on the following context: {context_text}\nAnswer the query: {query}"
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)
import google.generativeai as genai
import os

def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke the Gemini API given a system and user message.
    """
    # Configure Gemini API with your API key (set via environment variable or directly)
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))  # Ensure GEMINI_API_KEY is set in your environment
    model = genai.GenerativeModel("gemini-2.5-flash")  # Use gemini-2.5-flash for cost-efficiency

    # Combine system and user messages into a single prompt (Gemini doesn't use separate system/user roles)
    prompt = f"{system_message}\n\n{user_message}"
    
    # Call the Gemini API
    response = model.generate_content(prompt)
    return response.text.strip()
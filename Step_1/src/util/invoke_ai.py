import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke the Gemini API with retry logic.
    """
    try:
        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Combine system and user messages
        prompt = f"{system_message}\n\n{user_message}"
        
        # Add retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate response
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive wait: 5s, 10s, 15s
                    print(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                raise
                
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error: Unable to generate response. Please try again later."
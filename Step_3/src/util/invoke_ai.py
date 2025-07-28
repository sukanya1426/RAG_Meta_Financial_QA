import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
import re
import threading
from typing import Dict, Any
import logging

# Load environment variables
load_dotenv()

# Global rate limiting state
_api_lock = threading.Lock()
_last_request_time = 0
_request_count = 0
_quota_reset_time = 0

logger = logging.getLogger(__name__)

class QuotaManager:
    """Manages API quota and rate limiting"""
    
    def __init__(self):
        self.requests_per_minute = 15  # Conservative limit
        self.requests_per_day = 1500   # Conservative daily limit
        self.daily_count = 0
        self.minute_count = 0
        self.last_minute_reset = time.time()
        self.last_day_reset = time.time()
        
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting quota"""
        current_time = time.time()
        
        # Reset minute counter
        if current_time - self.last_minute_reset >= 60:
            self.minute_count = 0
            self.last_minute_reset = current_time
            
        # Reset daily counter
        if current_time - self.last_day_reset >= 86400:
            self.daily_count = 0
            self.last_day_reset = current_time
            
        return (self.minute_count < self.requests_per_minute and 
                self.daily_count < self.requests_per_day)
    
    def record_request(self):
        """Record that a request was made"""
        self.minute_count += 1
        self.daily_count += 1
        
    def get_wait_time(self) -> float:
        """Get recommended wait time before next request"""
        current_time = time.time()
        
        if self.minute_count >= self.requests_per_minute:
            return 60 - (current_time - self.last_minute_reset)
        
        # Add small delay between requests
        return 2.0

quota_manager = QuotaManager()

def extract_retry_delay(error_message: str) -> int:
    """Extract retry delay from Gemini error message"""
    try:
        # Look for retry_delay in error message
        match = re.search(r'retry_delay.*?seconds: (\d+)', error_message)
        if match:
            return int(match.group(1))
        return 30  # Default wait time
    except:
        return 30

def invoke_ai(system_message: str, user_message: str, max_retries: int = 5) -> str:
    """
    Generic function to invoke the Gemini API with advanced retry logic and quota management.
    """
    global _api_lock, _last_request_time, _request_count, _quota_reset_time
    
    with _api_lock:
        # Check quota before making request
        if not quota_manager.can_make_request():
            wait_time = quota_manager.get_wait_time()
            logger.warning(f"Rate limit approaching. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Minimum delay between requests
        current_time = time.time()
        time_since_last = current_time - _last_request_time
        if time_since_last < 1.0:  # Minimum 1 second between requests
            time.sleep(1.0 - time_since_last)
        
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
            
            # Add retry logic for rate limits and quota issues
            for attempt in range(max_retries):
                try:
                    # Record request attempt
                    quota_manager.record_request()
                    _last_request_time = time.time()
                    
                    # Generate response
                    response = model.generate_content(prompt)
                    
                    if response and response.text:
                        return response.text.strip()
                    else:
                        logger.warning("Empty response from Gemini API")
                        return "Error: Empty response from API"
                        
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"API error on attempt {attempt + 1}: {error_str}")
                    
                    # Handle different types of errors
                    if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                        if attempt < max_retries - 1:
                            # Extract retry delay from error message
                            retry_delay = extract_retry_delay(error_str)
                            wait_time = min(retry_delay + (attempt * 5), 60)  # Cap at 60 seconds
                            
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}). Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error("Max retries reached for rate limiting")
                            return "Error: API quota exceeded. Please try again later."
                    
                    elif "500" in error_str or "internal" in error_str.lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 3  # Progressive wait for server errors
                            logger.warning(f"Server error. Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                    
                    # For other errors, fail immediately
                    raise
                    
            # If we get here, all retries failed
            return "Error: Unable to generate response after multiple retries."
                
        except Exception as e:
            logger.error(f"Fatal error calling Gemini API: {e}")
            return f"Error: Unable to generate response. {str(e)}"
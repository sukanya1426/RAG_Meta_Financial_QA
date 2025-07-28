import re
from typing import Optional

def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from XML-like tags in text."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

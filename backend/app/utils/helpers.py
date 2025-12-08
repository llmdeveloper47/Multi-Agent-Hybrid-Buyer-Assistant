"""
Helper utility functions.
"""

import uuid
import re
from typing import Dict, Any, Optional


def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        Unique session identifier string
    """
    return str(uuid.uuid4())


def format_price(price: Optional[float]) -> str:
    """
    Format a price value for display.
    
    Args:
        price: Price value (can be None)
        
    Returns:
        Formatted price string
    """
    if price is None:
        return "N/A"
    
    if price >= 1_000_000:
        return f"${price / 1_000_000:.2f}M"
    elif price >= 1_000:
        return f"${price / 1_000:.0f}K"
    else:
        return f"${price:,.0f}"


def format_property_summary(prop: Dict[str, Any]) -> str:
    """
    Format a property dictionary into a readable summary.
    
    Args:
        prop: Property dictionary
        
    Returns:
        Formatted summary string
    """
    parts = []
    
    # Address
    address = prop.get("address", prop.get("Address"))
    if address:
        parts.append(address)
    
    # Location
    city = prop.get("city", prop.get("City"))
    state = prop.get("state", prop.get("State"))
    if city and state:
        parts.append(f"{city}, {state}")
    elif state:
        parts.append(state)
    
    # Details
    details = []
    
    price = prop.get("price", prop.get("Price"))
    if price:
        details.append(format_price(price))
    
    beds = prop.get("bedrooms", prop.get("Bedrooms"))
    if beds:
        details.append(f"{beds} bed")
    
    baths = prop.get("bathrooms", prop.get("Bathrooms"))
    if baths:
        details.append(f"{baths} bath")
    
    sqft = prop.get("sqft", prop.get("Size"))
    if sqft:
        details.append(f"{sqft:,.0f} sqft")
    
    if details:
        parts.append(" | ".join(details))
    
    return " - ".join(parts) if parts else "Unknown Property"


def parse_price_string(price_str: str) -> Optional[int]:
    """
    Parse a price string into an integer value.
    
    Handles formats like:
    - $500,000
    - 500k
    - 500K
    - $1.5M
    - 1500000
    
    Args:
        price_str: Price string to parse
        
    Returns:
        Integer price value or None if unparseable
    """
    if not price_str:
        return None
    
    # Clean the string
    price_str = price_str.strip().lower()
    price_str = price_str.replace('$', '').replace(',', '')
    
    try:
        # Check for millions (M)
        if 'm' in price_str:
            price_str = price_str.replace('m', '')
            return int(float(price_str) * 1_000_000)
        
        # Check for thousands (K)
        if 'k' in price_str:
            price_str = price_str.replace('k', '')
            return int(float(price_str) * 1_000)
        
        # Plain number
        value = float(price_str)
        
        # If value is small, assume it's in thousands
        if value < 10000:
            return int(value * 1000)
        
        return int(value)
        
    except (ValueError, TypeError):
        return None


def extract_numbers_from_text(text: str) -> list:
    """
    Extract all numbers from a text string.
    
    Args:
        text: Input text
        
    Returns:
        List of numbers found
    """
    pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # Remove commas and convert
            num = float(match.replace(',', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    return numbers


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_llm_response(response: str) -> str:
    """
    Clean up an LLM response by removing common artifacts.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Cleaned response
    """
    if not response:
        return ""
    
    # Remove leading/trailing whitespace
    response = response.strip()
    
    # Remove common artifacts
    artifacts = [
        "As an AI language model,",
        "As an AI assistant,",
        "I'd be happy to help!",
        "Certainly!",
        "Of course!",
    ]
    
    for artifact in artifacts:
        if response.startswith(artifact):
            response = response[len(artifact):].strip()
    
    return response


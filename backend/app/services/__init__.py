"""
Services for the Hybrid Buyer Advisor application.

- PropertySearchService: Vector search via Superlinked + Qdrant
- LLMService: Language model interactions
- SessionService: User session management
"""

from .llm_service import LLMService, get_llm_service
from .session_service import SessionService, get_session_service

# Import PropertySearchService from infrastructure
from ..infrastructure.superlinked import PropertySearchService, get_property_search_service

# Backward-compatible aliases
SuperlinkedService = PropertySearchService
get_superlinked_service = get_property_search_service

__all__ = [
    # New names
    "PropertySearchService",
    "get_property_search_service",
    # Backward-compatible aliases
    "SuperlinkedService", 
    "get_superlinked_service",
    # Other services
    "LLMService",
    "get_llm_service",
    "SessionService",
    "get_session_service",
]

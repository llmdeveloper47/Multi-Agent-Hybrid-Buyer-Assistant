"""
LangGraph state definitions for the Buyer Advisor workflow.
"""

from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    """Enumeration of possible user intents."""
    
    SEARCH_OR_RECOMMEND = "search_or_recommend"
    VALUATION = "valuation"
    COMPARISON = "comparison"
    MARKET_INSIGHT = "market_insight"
    FAVORITES = "favorites"
    UNKNOWN = "unknown"


class PropertyData(TypedDict, total=False):
    """Type definition for property data stored in state."""
    
    id: str
    property_type: str
    price: float
    bedrooms: int
    bathrooms: float
    sqft: float
    city: str
    state: str
    address: str
    zip_code: str
    description: str
    score: float  # Similarity score from vector search


class ConversationEntry(TypedDict):
    """Type definition for a conversation history entry."""
    
    role: str  # 'user' or 'assistant'
    content: str


class BuyerAdvisorState(TypedDict, total=False):
    """
    Main state object passed through the LangGraph workflow.
    
    This state is shared across all agent nodes and contains
    all the context needed for processing user queries.
    """
    
    # Input
    user_query: str
    session_id: str
    
    # Intent Classification
    intent: str  # One of IntentType values
    intent_confidence: float
    
    # Search/Filter Results
    results: List[PropertyData]
    search_filters: Dict[str, Any]
    
    # Valuation Data
    target_property: Optional[PropertyData]
    comparables: List[PropertyData]
    valuation_estimate: Dict[str, Any]
    market_stats: Dict[str, Any]
    
    # Comparison Data
    comparison_properties: List[PropertyData]
    comparison_result: str
    
    # Favorites Management
    favorites: List[PropertyData]
    favorite_action: str  # 'add', 'remove', 'list'
    favorite_target: Optional[str]  # Property ID to add/remove
    
    # Conversation Context
    history: List[ConversationEntry]
    last_shown_properties: List[PropertyData]  # Properties from last response
    
    # Output
    answer: str
    agent_output: str  # Raw output from specialist agent
    
    # Error Handling
    error: Optional[str]
    error_agent: Optional[str]


def create_initial_state(
    user_query: str,
    session_id: str,
    favorites: Optional[List[PropertyData]] = None,
    history: Optional[List[ConversationEntry]] = None,
    last_shown_properties: Optional[List[PropertyData]] = None
) -> BuyerAdvisorState:
    """
    Create an initial state object for a new query.
    
    Args:
        user_query: The user's query text
        session_id: Session identifier
        favorites: Existing favorites list from session
        history: Conversation history from session
        last_shown_properties: Properties from the last response
        
    Returns:
        Initialized BuyerAdvisorState
    """
    return BuyerAdvisorState(
        user_query=user_query,
        session_id=session_id,
        intent="",
        intent_confidence=0.0,
        results=[],
        search_filters={},
        target_property=None,
        comparables=[],
        valuation_estimate={},
        market_stats={},
        comparison_properties=[],
        comparison_result="",
        favorites=favorites or [],
        favorite_action="",
        favorite_target=None,
        history=history or [],
        last_shown_properties=last_shown_properties or [],
        answer="",
        agent_output="",
        error=None,
        error_agent=None
    )


def get_intent_type(intent_string: str) -> IntentType:
    """
    Convert an intent string to IntentType enum.
    
    Args:
        intent_string: String representation of intent
        
    Returns:
        Corresponding IntentType enum value
    """
    intent_map = {
        "search_or_recommend": IntentType.SEARCH_OR_RECOMMEND,
        "search": IntentType.SEARCH_OR_RECOMMEND,
        "recommend": IntentType.SEARCH_OR_RECOMMEND,
        "valuation": IntentType.VALUATION,
        "price": IntentType.VALUATION,
        "value": IntentType.VALUATION,
        "comparison": IntentType.COMPARISON,
        "compare": IntentType.COMPARISON,
        "market_insight": IntentType.MARKET_INSIGHT,
        "market": IntentType.MARKET_INSIGHT,
        "insight": IntentType.MARKET_INSIGHT,
        "favorites": IntentType.FAVORITES,
        "favorite": IntentType.FAVORITES,
        "save": IntentType.FAVORITES,
    }
    
    normalized = intent_string.lower().strip()
    return intent_map.get(normalized, IntentType.UNKNOWN)


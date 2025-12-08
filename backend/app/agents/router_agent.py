"""
Router Agent - Classifies user intent and routes to appropriate specialist agent.
"""

from typing import Dict, Any
import re

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState, IntentType, get_intent_type


class RouterAgent(BaseAgent):
    """
    Router Agent that classifies user intent and determines workflow path.
    
    Can use either rule-based classification or LLM-based classification.
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the Router Agent.
        
        Args:
            use_llm: Whether to use LLM for intent classification (default: rule-based)
        """
        super().__init__("router_agent")
        self.use_llm = use_llm
        
        # Keywords for rule-based classification
        self.intent_keywords = {
            IntentType.FAVORITES: [
                "favorite", "favourit", "save", "saved", "bookmark",
                "add to", "remove from", "show my", "list my", "my saved"
            ],
            IntentType.COMPARISON: [
                "compare", "comparison", "versus", "vs", "vs.", "difference",
                "better", "which one", "pros and cons", "trade-off", "tradeoff"
            ],
            IntentType.VALUATION: [
                "value", "valuation", "worth", "price estimate", "fair price",
                "good price", "overpriced", "underpriced", "market value",
                "how much should", "what should i pay", "is it worth"
            ],
            IntentType.MARKET_INSIGHT: [
                "market", "trend", "average price", "median price", "statistics",
                "stats", "how is the", "market condition", "buyer's market",
                "seller's market", "housing market"
            ],
            IntentType.SEARCH_OR_RECOMMEND: [
                "find", "search", "show", "looking for", "recommend", "suggest",
                "houses", "homes", "properties", "apartments", "condos",
                "bedroom", "bathroom", "under $", "around $", "near", "in"
            ]
        }
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Classify the user's intent and update state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with intent classification
        """
        user_query = state.get("user_query", "").lower()
        
        if self.use_llm:
            intent = self._classify_with_llm(user_query)
        else:
            intent = self._classify_with_rules(user_query)
        
        state["intent"] = intent
        state["intent_confidence"] = 1.0 if not self.use_llm else 0.8
        
        return state
    
    def _classify_with_rules(self, query: str) -> str:
        """
        Rule-based intent classification using keywords.
        
        Args:
            query: User query (lowercase)
            
        Returns:
            Intent string
        """
        query_lower = query.lower()
        
        # Check each intent category by priority
        # Favorites has highest priority (explicit user action)
        for intent_type in [
            IntentType.FAVORITES,
            IntentType.COMPARISON,
            IntentType.VALUATION,
            IntentType.MARKET_INSIGHT,
            IntentType.SEARCH_OR_RECOMMEND
        ]:
            keywords = self.intent_keywords.get(intent_type, [])
            for keyword in keywords:
                if keyword in query_lower:
                    return intent_type.value
        
        # Default to search/recommend
        return IntentType.SEARCH_OR_RECOMMEND.value
    
    def _classify_with_llm(self, query: str) -> str:
        """
        LLM-based intent classification.
        
        Args:
            query: User query
            
        Returns:
            Intent string
        """
        try:
            intent = self.llm.classify_intent(query)
            return intent
        except Exception as e:
            print(f"LLM classification failed, falling back to rules: {e}")
            return self._classify_with_rules(query)


# Create singleton instance
_router_agent: RouterAgent = None


def get_router_agent(use_llm: bool = False) -> RouterAgent:
    """Get or create the router agent singleton."""
    global _router_agent
    if _router_agent is None:
        _router_agent = RouterAgent(use_llm=use_llm)
    return _router_agent


def router_agent_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Router Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with intent
    """
    agent = get_router_agent()
    return agent.process(state)


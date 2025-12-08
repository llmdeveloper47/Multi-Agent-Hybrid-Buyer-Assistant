"""
Valuation Agent - Provides property value estimates and pricing analysis.
"""

from typing import Dict, Any, List, Optional
import re

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState


class ValuationAgent(BaseAgent):
    """
    Valuation Agent for property value estimates and pricing analysis.
    
    Uses comparable properties and market data to estimate fair market value.
    """
    
    def __init__(self):
        super().__init__("valuation_agent")
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process a valuation query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with valuation analysis
        """
        user_query = state.get("user_query", "")
        last_shown = state.get("last_shown_properties", [])
        favorites = state.get("favorites", [])
        
        # Try to identify the target property
        target_property = self._identify_target_property(
            user_query, last_shown, favorites
        )
        state["target_property"] = target_property
        
        # Get comparable properties
        comparables = self._get_comparables(target_property, user_query)
        state["comparables"] = comparables
        
        # Calculate market statistics
        market_stats = self._calculate_market_stats(comparables)
        state["market_stats"] = market_stats
        
        # Calculate valuation estimate
        if target_property and comparables:
            valuation = self._estimate_value(target_property, comparables)
            state["valuation_estimate"] = valuation
        
        # Generate response using LLM
        try:
            response = self.llm.generate_valuation_analysis(
                target_property=target_property,
                comparables=comparables,
                market_stats=market_stats,
                user_query=user_query
            )
            state["agent_output"] = response
            
        except Exception as e:
            state["error"] = f"Valuation analysis failed: {str(e)}"
            state["error_agent"] = self.agent_name
            state["agent_output"] = (
                "I apologize, but I couldn't complete the valuation analysis. "
                "Please provide more details about the property you'd like me to evaluate."
            )
        
        return state
    
    def _identify_target_property(
        self,
        query: str,
        last_shown: List[Dict],
        favorites: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Identify which property the user is asking about.
        
        Args:
            query: User query
            last_shown: Recently shown properties
            favorites: User's favorites
            
        Returns:
            Target property dict or None
        """
        query_lower = query.lower()
        
        # Check for ordinal references
        ordinal_map = {
            "first": 0, "1st": 0, "second": 1, "2nd": 1,
            "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4
        }
        
        for word, idx in ordinal_map.items():
            if word in query_lower and idx < len(last_shown):
                return last_shown[idx]
        
        # Check for "this" or "that" (most recent)
        if any(word in query_lower for word in ["this", "that", "it"]):
            if last_shown:
                return last_shown[0]
        
        # Check for address mentions
        all_properties = last_shown + favorites
        for prop in all_properties:
            address = prop.get("address", prop.get("Address", "")).lower()
            if address and any(part in query_lower for part in address.split()[:3]):
                return prop
        
        return None
    
    def _get_comparables(
        self,
        target: Optional[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Get comparable properties for valuation.
        
        Args:
            target: Target property (if identified)
            query: User query for context
            
        Returns:
            List of comparable properties
        """
        if target:
            # Search for similar properties
            search_query = self._build_comparable_query(target)
            filters = self._build_comparable_filters(target)
        else:
            # Use the query itself to find relevant properties
            search_query = query
            filters = None
        
        try:
            comparables = self.vector_store.search(
                query=search_query,
                top_k=10,
                filters=filters
            )
            
            # Filter out the target property if present
            if target:
                target_id = target.get("id")
                comparables = [c for c in comparables if c.get("id") != target_id]
            
            return comparables[:5]  # Return top 5 comparables
            
        except Exception:
            return []
    
    def _build_comparable_query(self, target: Dict[str, Any]) -> str:
        """Build a search query for comparable properties."""
        parts = []
        
        if target.get("property_type") or target.get("Type"):
            parts.append(target.get("property_type") or target.get("Type"))
        
        beds = target.get("bedrooms") or target.get("Bedrooms")
        if beds:
            parts.append(f"{beds} bedroom")
        
        city = target.get("city") or target.get("City")
        state = target.get("state") or target.get("State")
        if city and state:
            parts.append(f"in {city}, {state}")
        elif state:
            parts.append(f"in {state}")
        
        return " ".join(parts) if parts else "residential property"
    
    def _build_comparable_filters(self, target: Dict[str, Any]) -> Optional[Dict]:
        """Build filters for comparable search."""
        filters = {}
        
        # Same state
        state = target.get("state") or target.get("State")
        if state:
            filters["State"] = state
        
        # Similar bedroom count (+/- 1)
        beds = target.get("bedrooms") or target.get("Bedrooms")
        if beds:
            filters["Bedrooms"] = {"gte": max(1, beds - 1), "lte": beds + 1}
        
        # Similar price range (+/- 30%)
        price = target.get("price") or target.get("Price")
        if price:
            filters["Price"] = {
                "gte": int(price * 0.7),
                "lte": int(price * 1.3)
            }
        
        return filters if filters else None
    
    def _calculate_market_stats(self, comparables: List[Dict]) -> Dict[str, Any]:
        """
        Calculate market statistics from comparables.
        
        Args:
            comparables: List of comparable properties
            
        Returns:
            Dictionary of market statistics
        """
        if not comparables:
            return {}
        
        prices = []
        sqft_list = []
        price_per_sqft = []
        lot_sizes = []
        
        for comp in comparables:
            # Support both schemas
            price = comp.get("price") or comp.get("Price")
            sqft = comp.get("sqft") or comp.get("Size") or comp.get("house_size")
            acre_lot = comp.get("acre_lot") or comp.get("AcreLot")
            
            if price:
                prices.append(float(price))
            if sqft:
                sqft_list.append(float(sqft))
            if price and sqft and float(sqft) > 0:
                price_per_sqft.append(float(price) / float(sqft))
            if acre_lot:
                lot_sizes.append(float(acre_lot))
        
        stats = {}
        
        if prices:
            prices.sort()
            stats["min_price"] = min(prices)
            stats["max_price"] = max(prices)
            stats["avg_price"] = sum(prices) / len(prices)
            stats["median_price"] = prices[len(prices) // 2]
        
        if price_per_sqft:
            stats["avg_price_per_sqft"] = sum(price_per_sqft) / len(price_per_sqft)
        
        if lot_sizes:
            stats["avg_lot_size_acres"] = sum(lot_sizes) / len(lot_sizes)
        
        stats["comparable_count"] = len(comparables)
        
        return stats
    
    def _estimate_value(
        self,
        target: Dict[str, Any],
        comparables: List[Dict]
    ) -> Dict[str, Any]:
        """
        Estimate the value of a target property.
        
        Args:
            target: Target property
            comparables: Comparable properties
            
        Returns:
            Valuation estimate dictionary
        """
        market_stats = self._calculate_market_stats(comparables)
        
        if not market_stats:
            return {"estimate": None, "confidence": "low"}
        
        # Support both schemas for sqft
        target_sqft = target.get("sqft") or target.get("Size") or target.get("house_size")
        avg_price_sqft = market_stats.get("avg_price_per_sqft")
        
        if target_sqft and avg_price_sqft:
            target_sqft = float(target_sqft)
            # Estimate based on price per sqft
            estimated_value = target_sqft * avg_price_sqft
            
            # Create a range (+/- 10%)
            return {
                "estimated_value_low": int(estimated_value * 0.9),
                "estimated_value_mid": int(estimated_value),
                "estimated_value_high": int(estimated_value * 1.1),
                "avg_price_per_sqft": avg_price_sqft,
                "comparable_count": market_stats.get("comparable_count", 0),
                "confidence": "medium" if len(comparables) >= 3 else "low"
            }
        
        # Fallback: use average of comparables
        avg_price = market_stats.get("avg_price")
        if avg_price:
            return {
                "estimated_value_low": int(market_stats.get("min_price", avg_price * 0.9)),
                "estimated_value_mid": int(avg_price),
                "estimated_value_high": int(market_stats.get("max_price", avg_price * 1.1)),
                "comparable_count": market_stats.get("comparable_count", 0),
                "confidence": "low"
            }
        
        return {"estimate": None, "confidence": "very_low"}


# Singleton instance
_valuation_agent: ValuationAgent = None


def get_valuation_agent() -> ValuationAgent:
    """Get or create the valuation agent singleton."""
    global _valuation_agent
    if _valuation_agent is None:
        _valuation_agent = ValuationAgent()
    return _valuation_agent


def valuation_agent_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Valuation Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with valuation analysis
    """
    agent = get_valuation_agent()
    return agent.process(state)


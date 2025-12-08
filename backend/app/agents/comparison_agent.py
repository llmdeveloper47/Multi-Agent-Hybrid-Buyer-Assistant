"""
Comparison Agent - Provides trade-off analysis and property comparisons.
"""

from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState


class ComparisonAgent(BaseAgent):
    """
    Comparison Agent for property comparisons and trade-off analysis.
    
    Compares multiple properties side-by-side and provides pros/cons.
    """
    
    def __init__(self):
        super().__init__("comparison_agent")
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process a comparison query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with comparison analysis
        """
        user_query = state.get("user_query", "")
        last_shown = state.get("last_shown_properties", [])
        favorites = state.get("favorites", [])
        
        # Identify properties to compare
        properties_to_compare = self._identify_comparison_targets(
            user_query, last_shown, favorites
        )
        state["comparison_properties"] = properties_to_compare
        
        # Extract user preferences if mentioned
        preferences = self._extract_preferences(user_query)
        
        # Generate comparison using LLM
        try:
            if properties_to_compare:
                response = self.llm.generate_comparison(
                    properties=properties_to_compare,
                    user_query=user_query,
                    preferences=preferences
                )
            else:
                # Handle general comparison questions (e.g., "condo vs house")
                response = self._handle_general_comparison(user_query)
            
            state["agent_output"] = response
            state["comparison_result"] = response
            
        except Exception as e:
            state["error"] = f"Comparison failed: {str(e)}"
            state["error_agent"] = self.agent_name
            state["agent_output"] = (
                "I apologize, but I couldn't complete the comparison. "
                "Please specify which properties you'd like me to compare."
            )
        
        return state
    
    def _identify_comparison_targets(
        self,
        query: str,
        last_shown: List[Dict],
        favorites: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Identify which properties the user wants to compare.
        
        Args:
            query: User query
            last_shown: Recently shown properties
            favorites: User's favorites
            
        Returns:
            List of properties to compare
        """
        query_lower = query.lower()
        properties = []
        
        # Check if comparing favorites
        if "favorite" in query_lower or "saved" in query_lower:
            if len(favorites) >= 2:
                return favorites[:5]  # Compare up to 5 favorites
        
        # Check for specific ordinal references
        ordinal_map = {
            "first": 0, "1st": 0, "one": 0, "1": 0,
            "second": 1, "2nd": 1, "two": 1, "2": 1,
            "third": 2, "3rd": 2, "three": 2, "3": 2,
            "fourth": 3, "4th": 3, "four": 3, "4": 3,
            "fifth": 4, "5th": 4, "five": 4, "5": 4,
        }
        
        found_indices = set()
        for word, idx in ordinal_map.items():
            if word in query_lower and idx < len(last_shown):
                found_indices.add(idx)
        
        if found_indices:
            for idx in sorted(found_indices):
                if idx < len(last_shown):
                    properties.append(last_shown[idx])
            return properties
        
        # Check for "last two", "all", etc.
        if "last two" in query_lower or "both" in query_lower:
            return last_shown[:2] if len(last_shown) >= 2 else last_shown
        
        if "all" in query_lower:
            return last_shown[:5] if len(last_shown) > 0 else favorites[:5]
        
        # Check for address mentions
        all_properties = last_shown + favorites
        for prop in all_properties:
            address = prop.get("address", prop.get("Address", "")).lower()
            city = prop.get("city", prop.get("City", "")).lower()
            
            if address or city:
                # Check if property is mentioned
                if (address and any(part in query_lower for part in address.split()[:2])) or \
                   (city and city in query_lower):
                    if prop not in properties:
                        properties.append(prop)
        
        # If we found some but not enough, add from last shown
        if 1 <= len(properties) < 2 and last_shown:
            for prop in last_shown:
                if prop not in properties:
                    properties.append(prop)
                    if len(properties) >= 2:
                        break
        
        # Default: compare last two shown properties
        if not properties and len(last_shown) >= 2:
            return last_shown[:2]
        
        return properties
    
    def _extract_preferences(self, query: str) -> Dict[str, Any]:
        """
        Extract user preferences from the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted preferences
        """
        preferences = {}
        query_lower = query.lower()
        
        # Priority keywords
        priority_keywords = {
            "price": ["budget", "price", "cheap", "affordable", "cost"],
            "space": ["space", "room", "big", "large", "spacious", "sqft"],
            "location": ["location", "commute", "near", "close to", "neighborhood"],
            "family": ["family", "kids", "children", "school"],
            "modern": ["modern", "new", "updated", "renovated"],
        }
        
        for pref_type, keywords in priority_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    preferences[pref_type] = True
                    break
        
        return preferences if preferences else None
    
    def _handle_general_comparison(self, query: str) -> str:
        """
        Handle general comparison questions (e.g., condo vs house).
        
        Args:
            query: User query
            
        Returns:
            General comparison response
        """
        # Use LLM for general real estate knowledge
        prompt = f"""The user is asking a general real estate comparison question:
        
        Query: {query}
        
        Please provide a helpful comparison with pros and cons for each option.
        Focus on practical considerations for home buyers.
        """
        
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt
        )


# Singleton instance
_comparison_agent: ComparisonAgent = None


def get_comparison_agent() -> ComparisonAgent:
    """Get or create the comparison agent singleton."""
    global _comparison_agent
    if _comparison_agent is None:
        _comparison_agent = ComparisonAgent()
    return _comparison_agent


def comparison_agent_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Comparison Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with comparison analysis
    """
    agent = get_comparison_agent()
    return agent.process(state)


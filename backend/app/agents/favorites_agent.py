"""
Favorites Agent - Manages user's saved/favorite properties.
"""

from typing import Dict, Any, List, Optional
import re

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState


class FavoritesAgent(BaseAgent):
    """
    Favorites Agent for managing user's saved properties.
    
    Handles adding, removing, and listing favorite properties.
    """
    
    def __init__(self):
        super().__init__("favorites_agent")
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process a favorites management query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with favorites changes
        """
        user_query = state.get("user_query", "")
        favorites = state.get("favorites", [])
        last_shown = state.get("last_shown_properties", [])
        
        # Determine the action
        action = self._determine_action(user_query)
        state["favorite_action"] = action
        
        if action == "add":
            result = self._handle_add(user_query, favorites, last_shown, state)
        elif action == "remove":
            result = self._handle_remove(user_query, favorites, state)
        elif action == "list":
            result = self._handle_list(favorites)
        else:
            result = self._handle_unknown(user_query, favorites, last_shown)
        
        state["agent_output"] = result
        
        return state
    
    def _determine_action(self, query: str) -> str:
        """
        Determine what action the user wants to perform.
        
        Args:
            query: User query
            
        Returns:
            Action string: 'add', 'remove', 'list', or 'unknown'
        """
        query_lower = query.lower()
        
        # Add actions
        add_keywords = ["add", "save", "bookmark", "favorite this", "keep"]
        if any(kw in query_lower for kw in add_keywords):
            return "add"
        
        # Remove actions
        remove_keywords = ["remove", "delete", "unfavorite", "unsave"]
        if any(kw in query_lower for kw in remove_keywords):
            return "remove"
        
        # List actions
        list_keywords = ["show", "list", "what are my", "see my", "my favorites", "my saved"]
        if any(kw in query_lower for kw in list_keywords):
            return "list"
        
        return "unknown"
    
    def _handle_add(
        self,
        query: str,
        favorites: List[Dict],
        last_shown: List[Dict],
        state: BuyerAdvisorState
    ) -> str:
        """
        Handle adding a property to favorites.
        
        Args:
            query: User query
            favorites: Current favorites list
            last_shown: Recently shown properties
            state: Current state (will be modified)
            
        Returns:
            Response message
        """
        # Identify which property to add
        target_property = self._identify_property(query, last_shown, favorites)
        
        if not target_property:
            # If no specific property identified, try the most recent
            if last_shown:
                target_property = last_shown[0]
            else:
                return (
                    "I'm not sure which property you'd like to save. "
                    "Could you specify by saying something like 'add the first one' or 'save the house on Maple Street'?"
                )
        
        # Check if already in favorites
        target_id = target_property.get("id")
        if any(f.get("id") == target_id for f in favorites):
            address = target_property.get("address", target_property.get("Address", "This property"))
            return f"{address} is already in your favorites!"
        
        # Add to favorites
        favorites.append(target_property)
        state["favorites"] = favorites
        state["favorite_target"] = target_id
        
        # Generate response
        address = target_property.get("address", target_property.get("Address", "The property"))
        price = target_property.get("price", target_property.get("Price"))
        beds = target_property.get("bedrooms", target_property.get("Bedrooms"))
        
        response = f"✓ Added to your favorites: {address}"
        if price:
            response += f" (${price:,.0f}"
            if beds:
                response += f", {beds} bed"
            response += ")"
        
        response += f"\n\nYou now have {len(favorites)} property(ies) saved."
        
        return response
    
    def _handle_remove(
        self,
        query: str,
        favorites: List[Dict],
        state: BuyerAdvisorState
    ) -> str:
        """
        Handle removing a property from favorites.
        
        Args:
            query: User query
            favorites: Current favorites list
            state: Current state (will be modified)
            
        Returns:
            Response message
        """
        if not favorites:
            return "Your favorites list is empty. There's nothing to remove."
        
        # Identify which property to remove
        target_property = self._identify_property(query, favorites, favorites)
        
        if not target_property:
            return (
                "I'm not sure which property you'd like to remove. "
                "Please specify by address or position (e.g., 'remove the first one')."
            )
        
        # Remove from favorites
        target_id = target_property.get("id")
        original_count = len(favorites)
        state["favorites"] = [f for f in favorites if f.get("id") != target_id]
        state["favorite_target"] = target_id
        
        if len(state["favorites"]) < original_count:
            address = target_property.get("address", target_property.get("Address", "The property"))
            return f"✓ Removed from favorites: {address}\n\nYou have {len(state['favorites'])} favorite(s) remaining."
        else:
            return "I couldn't find that property in your favorites."
    
    def _handle_list(self, favorites: List[Dict]) -> str:
        """
        Handle listing all favorites.
        
        Args:
            favorites: Current favorites list
            
        Returns:
            Formatted list of favorites
            
        Supports both realtor-data schema and standard schema.
        """
        if not favorites:
            return (
                "You haven't saved any favorites yet. "
                "When you find properties you like, just say 'add to favorites' to save them!"
            )
        
        response = f"📋 **Your Favorites ({len(favorites)})**\n\n"
        
        for i, fav in enumerate(favorites, 1):
            # Support both schemas
            address = fav.get("address") or fav.get("Address") or fav.get("street") or "Unknown"
            city = fav.get("city") or fav.get("City") or ""
            state = fav.get("state") or fav.get("State") or ""
            price = fav.get("price") or fav.get("Price")
            beds = fav.get("bedrooms") or fav.get("Bedrooms") or fav.get("bed")
            baths = fav.get("bathrooms") or fav.get("Bathrooms") or fav.get("bath")
            sqft = fav.get("sqft") or fav.get("Size") or fav.get("house_size")
            acre_lot = fav.get("acre_lot") or fav.get("AcreLot")
            
            response += f"{i}. **{address}**"
            if city or state:
                response += f" ({city}, {state})" if city else f" ({state})"
            response += "\n"
            
            details = []
            if price:
                details.append(f"${float(price):,.0f}")
            if beds:
                details.append(f"{int(beds)} bed")
            if baths:
                details.append(f"{float(baths):.1f} bath")
            if sqft:
                details.append(f"{float(sqft):,.0f} sqft")
            if acre_lot:
                details.append(f"{float(acre_lot):.2f} acres")
            
            if details:
                response += f"   {' | '.join(details)}\n"
            response += "\n"
        
        response += "\n💡 You can say 'compare my favorites' to see them side by side."
        
        return response
    
    def _handle_unknown(
        self,
        query: str,
        favorites: List[Dict],
        last_shown: List[Dict]
    ) -> str:
        """Handle unknown favorites action."""
        try:
            response = self.llm.generate_favorites_response(
                favorites=favorites,
                recent_properties=last_shown,
                user_query=query
            )
            return response
        except Exception:
            return (
                "I can help you manage your favorites. You can:\n"
                "- 'Add to favorites' - save a property\n"
                "- 'Remove from favorites' - remove a saved property\n"
                "- 'Show my favorites' - see all saved properties\n"
                "- 'Compare my favorites' - compare saved properties"
            )
    
    def _identify_property(
        self,
        query: str,
        search_list: List[Dict],
        favorites: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Identify which property the user is referring to.
        
        Args:
            query: User query
            search_list: Primary list to search
            favorites: Favorites list as fallback
            
        Returns:
            Property dictionary or None
        """
        query_lower = query.lower()
        
        # Check ordinal references
        ordinal_map = {
            "first": 0, "1st": 0, "one": 0, "1": 0,
            "second": 1, "2nd": 1, "two": 1, "2": 1,
            "third": 2, "3rd": 2, "three": 2, "3": 3,
            "fourth": 3, "4th": 3, "four": 3, "4": 3,
            "fifth": 4, "5th": 4, "five": 4, "5": 5,
            "last": -1,
        }
        
        for word, idx in ordinal_map.items():
            if word in query_lower:
                if idx == -1:
                    idx = len(search_list) - 1
                if 0 <= idx < len(search_list):
                    return search_list[idx]
        
        # Check for "this", "that", "it" (most recent)
        if any(word in query_lower for word in ["this", "that", "it"]):
            if search_list:
                return search_list[0]
        
        # Check for address mentions
        all_properties = search_list + favorites
        for prop in all_properties:
            address = prop.get("address", prop.get("Address", "")).lower()
            if address:
                # Check if significant part of address is in query
                address_parts = address.split()[:3]  # First 3 words
                if any(part in query_lower for part in address_parts if len(part) > 2):
                    return prop
        
        return None


# Singleton instance
_favorites_agent: FavoritesAgent = None


def get_favorites_agent() -> FavoritesAgent:
    """Get or create the favorites agent singleton."""
    global _favorites_agent
    if _favorites_agent is None:
        _favorites_agent = FavoritesAgent()
    return _favorites_agent


def favorites_agent_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Favorites Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with favorites changes
    """
    agent = get_favorites_agent()
    return agent.process(state)


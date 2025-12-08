"""
Market Insights Agent - Provides market information and statistics.
"""

from typing import Dict, Any, List, Optional
import re

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState


class MarketInsightsAgent(BaseAgent):
    """
    Market Insights Agent for providing real estate market information.
    
    Provides statistics, trends, and general market knowledge.
    """
    
    def __init__(self):
        super().__init__("market_insights_agent")
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process a market insight query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with market insights
        """
        user_query = state.get("user_query", "")
        
        # Identify the area/location of interest
        area = self._extract_location(user_query)
        
        # Get market statistics for the area
        statistics = self._get_market_statistics(area, user_query)
        state["market_stats"] = statistics
        
        # Generate response using LLM
        try:
            response = self.llm.generate_market_insight(
                statistics=statistics,
                area=area or "general market",
                user_query=user_query
            )
            state["agent_output"] = response
            
        except Exception as e:
            state["error"] = f"Market insight generation failed: {str(e)}"
            state["error_agent"] = self.agent_name
            state["agent_output"] = (
                "I apologize, but I couldn't retrieve the market information you requested. "
                "Please try asking about a specific city or state."
            )
        
        return state
    
    def _extract_location(self, query: str) -> Optional[str]:
        """
        Extract location from user query.
        
        Args:
            query: User query
            
        Returns:
            Location string or None
        """
        query_lower = query.lower()
        
        # Common patterns for location extraction
        patterns = [
            r'in\s+([A-Za-z\s]+?)(?:\s+market|\?|$|,)',
            r'(?:for|of)\s+([A-Za-z\s]+?)(?:\s+market|\?|$|,)',
            r'([A-Za-z\s]+?)\s+(?:housing|real estate|market)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                # Clean up common words
                location = re.sub(
                    r'\b(the|housing|real estate|market|prices?|average|median)\b',
                    '', location
                ).strip()
                if location and len(location) > 1:
                    return location.title()
        
        # Check for state abbreviations
        state_pattern = r'\b([A-Z]{2})\b'
        match = re.search(state_pattern, query)
        if match:
            return match.group(1)
        
        return None
    
    def _get_market_statistics(
        self,
        area: Optional[str],
        query: str
    ) -> Dict[str, Any]:
        """
        Get market statistics for an area.
        
        Args:
            area: Location/area of interest
            query: User query for context
            
        Returns:
            Dictionary of market statistics
        """
        statistics = {}
        
        # Build filter for the area
        filters = self._build_area_filter(area) if area else None
        
        try:
            # Get sample of properties from the area
            results = self.vector_store.search(
                query=f"properties in {area}" if area else "residential properties",
                top_k=100,  # Get more for statistics
                filters=filters
            )
            
            if results:
                statistics = self._calculate_statistics(results, area)
            
        except Exception:
            pass
        
        # Add general market information
        statistics["data_source"] = "Property listing database"
        statistics["note"] = "Statistics based on available listing data"
        
        return statistics
    
    def _build_area_filter(self, area: str) -> Optional[Dict]:
        """Build filter for area-based search."""
        if not area:
            return None
        
        # State abbreviation to full name mapping (for realtor-data schema)
        state_abbrev_to_full = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands'
        }
        
        # Full name to abbreviation (reverse lookup)
        state_full_to_abbrev = {v.lower(): k for k, v in state_abbrev_to_full.items()}
        
        area_upper = area.upper()
        area_lower = area.lower()
        area_title = area.title()
        
        # Check if it's a state abbreviation
        if area_upper in state_abbrev_to_full:
            # Return full state name for realtor-data schema
            return {"State": state_abbrev_to_full[area_upper]}
        
        # Check if it's a full state name
        if area_lower in state_full_to_abbrev:
            return {"State": area_title}
        
        # Partial match for state names (e.g., "puerto" matches "Puerto Rico")
        for full_name in state_abbrev_to_full.values():
            if area_lower in full_name.lower():
                return {"State": full_name}
        
        # For cities, rely on semantic search
        return None
    
    def _calculate_statistics(
        self,
        properties: List[Dict],
        area: Optional[str]
    ) -> Dict[str, Any]:
        """
        Calculate market statistics from property data.
        
        Args:
            properties: List of property dictionaries
            area: Area name for labeling
            
        Returns:
            Statistics dictionary
            
        Supports both realtor-data schema and standard schema.
        """
        stats = {"area": area or "All Areas"}
        
        prices = []
        sqft_list = []
        bedrooms = []
        bathrooms = []
        price_per_sqft = []
        lot_sizes = []
        
        property_types = {}
        
        for prop in properties:
            # Support both schemas
            price = prop.get("price") or prop.get("Price")
            sqft = prop.get("sqft") or prop.get("Size") or prop.get("house_size")
            beds = prop.get("bedrooms") or prop.get("Bedrooms") or prop.get("bed")
            baths = prop.get("bathrooms") or prop.get("Bathrooms") or prop.get("bath")
            ptype = prop.get("property_type") or prop.get("Type") or prop.get("status")
            acre_lot = prop.get("acre_lot") or prop.get("AcreLot")
            
            if price and float(price) > 0:
                prices.append(float(price))
            if sqft and float(sqft) > 0:
                sqft_list.append(float(sqft))
            if beds:
                bedrooms.append(float(beds))
            if baths:
                bathrooms.append(float(baths))
            if price and sqft and float(sqft) > 0:
                price_per_sqft.append(float(price) / float(sqft))
            if ptype:
                ptype_str = str(ptype).replace('_', ' ').title()
                property_types[ptype_str] = property_types.get(ptype_str, 0) + 1
            if acre_lot:
                lot_sizes.append(float(acre_lot))
        
        # Price statistics
        if prices:
            prices.sort()
            n = len(prices)
            stats["total_listings"] = n
            stats["min_price"] = min(prices)
            stats["max_price"] = max(prices)
            stats["average_price"] = sum(prices) / n
            stats["median_price"] = prices[n // 2]
            
            # Percentiles
            stats["price_25th_percentile"] = prices[int(n * 0.25)]
            stats["price_75th_percentile"] = prices[int(n * 0.75)]
        
        # Price per sqft
        if price_per_sqft:
            stats["avg_price_per_sqft"] = sum(price_per_sqft) / len(price_per_sqft)
        
        # Size statistics
        if sqft_list:
            stats["avg_sqft"] = sum(sqft_list) / len(sqft_list)
        
        # Lot size statistics
        if lot_sizes:
            stats["avg_lot_size_acres"] = sum(lot_sizes) / len(lot_sizes)
        
        # Bedroom/bathroom averages
        if bedrooms:
            stats["avg_bedrooms"] = sum(bedrooms) / len(bedrooms)
        if bathrooms:
            stats["avg_bathrooms"] = sum(bathrooms) / len(bathrooms)
        
        # Property type breakdown
        if property_types:
            stats["property_types"] = property_types
        
        return stats


# Singleton instance
_market_insights_agent: MarketInsightsAgent = None


def get_market_insights_agent() -> MarketInsightsAgent:
    """Get or create the market insights agent singleton."""
    global _market_insights_agent
    if _market_insights_agent is None:
        _market_insights_agent = MarketInsightsAgent()
    return _market_insights_agent


def market_insights_agent_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Market Insights Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with market insights
    """
    agent = get_market_insights_agent()
    return agent.process(state)


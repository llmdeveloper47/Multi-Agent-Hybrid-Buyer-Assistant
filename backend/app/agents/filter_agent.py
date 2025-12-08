"""
Filter & Matching Agent - Handles property search and recommendations.
"""

from typing import Dict, Any, List, Optional
import re

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState


class FilterAgent(BaseAgent):
    """
    Filter & Matching Agent for property search and recommendations.
    
    Uses vector search for semantic matching and metadata filters for
    structured criteria (price, bedrooms, location, etc.).
    """
    
    def __init__(self):
        super().__init__("filter_agent")
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process a search/recommendation query.
        
        Uses Superlinked with Qdrant for natural language property search.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with search results
        """
        user_query = state.get("user_query", "")
        history = state.get("history", [])
        
        # Extract structured filters from query
        filters = self._extract_filters(user_query)
        state["search_filters"] = filters
        
        # Perform search using Superlinked with natural language query
        try:
            results = self.superlinked.search(
                query=user_query,
                top_k=self._settings.SEARCH_TOP_K,
                filters=self._build_superlinked_filters(filters),
                use_natural_language=True  # Enable LLM-powered query understanding
            )
            
            state["results"] = results
            
            # Generate natural language response
            response = self.llm.generate_property_summary(
                properties=results,
                user_query=user_query,
                history=history
            )
            state["agent_output"] = response
            
        except Exception as e:
            state["error"] = f"Search failed: {str(e)}"
            state["error_agent"] = self.agent_name
            state["agent_output"] = (
                "I apologize, but I encountered an issue while searching for properties. "
                "Please try rephrasing your search or check back later."
            )
        
        return state
    
    def _build_superlinked_filters(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Superlinked-compatible filters.
        
        Args:
            filters: Extracted filter dictionary
            
        Returns:
            Superlinked filter dict
        """
        if not filters:
            return None
        
        sl_filters = {}
        
        if 'min_price' in filters:
            sl_filters['min_price'] = filters['min_price']
        if 'max_price' in filters:
            sl_filters['max_price'] = filters['max_price']
        if 'bedrooms' in filters:
            sl_filters['min_bedrooms'] = filters['bedrooms']
            sl_filters['target_bedrooms'] = filters['bedrooms']
        if 'bathrooms' in filters:
            sl_filters['target_bathrooms'] = filters['bathrooms']
        if 'location' in filters:
            # State filter - use full state name for Superlinked
            location = filters['location']
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
            
            location_upper = location.upper()
            if location_upper in state_abbrev_to_full:
                sl_filters['state'] = state_abbrev_to_full[location_upper]
            elif location.title() in state_abbrev_to_full.values():
                sl_filters['state'] = location.title()
            else:
                # Could be a city
                sl_filters['city'] = location.title()
        
        return sl_filters if sl_filters else None
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract structured filters from natural language query.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary of extracted filters
        """
        filters = {}
        query_lower = query.lower()
        
        # Extract price constraints
        price_patterns = [
            (r'under\s*\$?([\d,]+)k?\b', 'max_price'),
            (r'below\s*\$?([\d,]+)k?\b', 'max_price'),
            (r'less than\s*\$?([\d,]+)k?\b', 'max_price'),
            (r'above\s*\$?([\d,]+)k?\b', 'min_price'),
            (r'over\s*\$?([\d,]+)k?\b', 'min_price'),
            (r'more than\s*\$?([\d,]+)k?\b', 'min_price'),
            (r'around\s*\$?([\d,]+)k?\b', 'target_price'),
            (r'about\s*\$?([\d,]+)k?\b', 'target_price'),
            (r'\$?([\d,]+)k?\s*(?:to|-)\s*\$?([\d,]+)k?\b', 'price_range'),
        ]
        
        for pattern, filter_type in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if filter_type == 'price_range':
                    min_price = self._parse_price(match.group(1))
                    max_price = self._parse_price(match.group(2))
                    filters['min_price'] = min_price
                    filters['max_price'] = max_price
                elif filter_type == 'target_price':
                    target = self._parse_price(match.group(1))
                    filters['min_price'] = int(target * 0.85)
                    filters['max_price'] = int(target * 1.15)
                else:
                    filters[filter_type] = self._parse_price(match.group(1))
                break
        
        # Extract bedroom count
        bed_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br|bd)',
            r'(\d+)\s*-?\s*(?:bed|bedroom)',
        ]
        for pattern in bed_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters['bedrooms'] = int(match.group(1))
                break
        
        # Extract bathroom count
        bath_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)',
        ]
        for pattern in bath_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters['bathrooms'] = float(match.group(1))
                break
        
        # Extract location (city/state) - simplified pattern
        # In production, you'd use a more sophisticated NER approach
        location_patterns = [
            r'in\s+([A-Za-z\s]+?)(?:\s+under|\s+around|\s+with|$|,)',
            r'near\s+([A-Za-z\s]+?)(?:\s+under|\s+around|\s+with|$|,)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                # Clean up common words
                location = re.sub(r'\b(houses?|homes?|properties|apartments?|condos?)\b', '', location).strip()
                if location:
                    filters['location'] = location
                break
        
        # Extract property type
        property_types = {
            'house': 'Single Family',
            'single family': 'Single Family',
            'condo': 'Condo',
            'condominium': 'Condo',
            'apartment': 'Apartment',
            'townhouse': 'Townhouse',
            'townhome': 'Townhouse',
            'multi-family': 'Multi-Family',
            'multifamily': 'Multi-Family',
        }
        for keyword, prop_type in property_types.items():
            if keyword in query_lower:
                filters['property_type'] = prop_type
                break
        
        return filters
    
    def _parse_price(self, price_str: str) -> int:
        """Parse a price string to integer."""
        # Remove commas
        price_str = price_str.replace(',', '')
        price = int(float(price_str))
        
        # Handle 'k' notation (e.g., 500k = 500,000)
        if price < 10000:  # Likely in thousands
            price *= 1000
        
        return price


# Singleton instance
_filter_agent: FilterAgent = None


def get_filter_agent() -> FilterAgent:
    """Get or create the filter agent singleton."""
    global _filter_agent
    if _filter_agent is None:
        _filter_agent = FilterAgent()
    return _filter_agent


def filter_agent_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Filter Agent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with search results
    """
    agent = get_filter_agent()
    return agent.process(state)


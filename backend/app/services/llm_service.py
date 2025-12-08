"""
LLM service for interacting with OpenAI GPT models.
"""

from typing import Optional, List, Dict, Any
from functools import lru_cache

from ..config import get_settings, load_system_prompt


class LLMService:
    """
    Service for interacting with Large Language Models (OpenAI GPT).
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
            print(f"Initialized OpenAI LLM client with model: {self.settings.OPENAI_MODEL}")
            
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt (uses default if not provided)
            temperature: Sampling temperature (uses settings default if not provided)
            max_tokens: Maximum tokens in response (uses settings default if not provided)
            model: Model to use (uses settings default if not provided)
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=model or self.settings.OPENAI_MODEL,
            messages=messages,
            temperature=temperature or self.settings.OPENAI_TEMPERATURE,
            max_tokens=max_tokens or self.settings.OPENAI_MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    def generate_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response with conversation history context.
        
        Args:
            prompt: The current user prompt
            history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for entry in history[-10:]:  # Limit to last 10 messages for context window
            messages.append({
                "role": entry.get("role", "user"),
                "content": entry.get("content", "")
            })
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.settings.OPENAI_MODEL,
            messages=messages,
            temperature=temperature or self.settings.OPENAI_TEMPERATURE,
            max_tokens=max_tokens or self.settings.OPENAI_MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    def classify_intent(self, user_query: str) -> str:
        """
        Classify the intent of a user query.
        
        Args:
            user_query: The user's query text
            
        Returns:
            Intent classification string
        """
        prompt_template = load_system_prompt("router_agent")
        prompt = prompt_template.format(user_query=user_query)
        
        # Use lower temperature for classification
        response = self.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=50
        )
        
        # Parse the response to extract intent
        response_lower = response.lower().strip()
        
        intent_keywords = {
            "search_or_recommend": ["search_or_recommend", "search", "recommend", "find"],
            "valuation": ["valuation", "price", "value", "worth"],
            "comparison": ["comparison", "compare"],
            "market_insight": ["market_insight", "market", "insight"],
            "favorites": ["favorites", "favorite", "save"]
        }
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    return intent
        
        # Default to search if unclear
        return "search_or_recommend"
    
    def generate_property_summary(
        self,
        properties: List[Dict[str, Any]],
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a natural language summary of properties.
        
        Args:
            properties: List of property dictionaries
            user_query: The user's original query
            history: Conversation history
            
        Returns:
            Natural language summary
        """
        prompt_template = load_system_prompt("filter_agent")
        
        # Format properties for the prompt
        properties_text = self._format_properties_for_prompt(properties)
        history_text = self._format_history_for_prompt(history) if history else "No previous conversation."
        
        prompt = prompt_template.format(
            user_query=user_query,
            properties=properties_text,
            history=history_text
        )
        
        return self.generate(prompt=prompt)
    
    def generate_valuation_analysis(
        self,
        target_property: Optional[Dict[str, Any]],
        comparables: List[Dict[str, Any]],
        market_stats: Dict[str, Any],
        user_query: str
    ) -> str:
        """
        Generate a valuation analysis.
        
        Args:
            target_property: The property being valued (if specific)
            comparables: List of comparable properties
            market_stats: Market statistics for the area
            user_query: The user's query
            
        Returns:
            Valuation analysis text
        """
        prompt_template = load_system_prompt("valuation_agent")
        
        target_text = self._format_property_for_prompt(target_property) if target_property else "Not specified"
        comparables_text = self._format_properties_for_prompt(comparables)
        stats_text = self._format_stats_for_prompt(market_stats)
        
        prompt = prompt_template.format(
            user_query=user_query,
            target_property=target_text,
            comparables=comparables_text,
            market_stats=stats_text
        )
        
        return self.generate(prompt=prompt)
    
    def generate_comparison(
        self,
        properties: List[Dict[str, Any]],
        user_query: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a property comparison.
        
        Args:
            properties: Properties to compare
            user_query: The user's query
            preferences: User preferences if known
            
        Returns:
            Comparison analysis text
        """
        prompt_template = load_system_prompt("comparison_agent")
        
        properties_text = self._format_properties_for_prompt(properties)
        preferences_text = str(preferences) if preferences else "Not specified"
        
        prompt = prompt_template.format(
            user_query=user_query,
            properties=properties_text,
            preferences=preferences_text
        )
        
        return self.generate(prompt=prompt)
    
    def generate_market_insight(
        self,
        statistics: Dict[str, Any],
        area: str,
        user_query: str
    ) -> str:
        """
        Generate market insight response.
        
        Args:
            statistics: Market statistics
            area: Geographic area
            user_query: The user's query
            
        Returns:
            Market insight text
        """
        prompt_template = load_system_prompt("market_insights_agent")
        
        stats_text = self._format_stats_for_prompt(statistics)
        
        prompt = prompt_template.format(
            user_query=user_query,
            statistics=stats_text,
            area=area
        )
        
        return self.generate(prompt=prompt)
    
    def generate_favorites_response(
        self,
        favorites: List[Dict[str, Any]],
        recent_properties: List[Dict[str, Any]],
        user_query: str
    ) -> str:
        """
        Generate response for favorites management.
        
        Args:
            favorites: Current favorites list
            recent_properties: Recently shown properties
            user_query: The user's query
            
        Returns:
            Favorites response text
        """
        prompt_template = load_system_prompt("favorites_agent")
        
        favorites_text = self._format_properties_for_prompt(favorites) if favorites else "No favorites saved yet."
        recent_text = self._format_properties_for_prompt(recent_properties) if recent_properties else "None"
        
        prompt = prompt_template.format(
            user_query=user_query,
            favorites=favorites_text,
            recent_properties=recent_text
        )
        
        return self.generate(prompt=prompt)
    
    def compose_final_response(
        self,
        agent_output: str,
        user_query: str,
        intent: str,
        history: Optional[List[Dict[str, str]]] = None,
        favorites_count: int = 0,
        last_properties: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Compose the final response using the conversation manager prompt.
        
        Args:
            agent_output: Output from the specialist agent
            user_query: The user's original query
            intent: Detected intent
            history: Conversation history
            favorites_count: Number of favorites
            last_properties: Last shown properties
            
        Returns:
            Final composed response
        """
        prompt_template = load_system_prompt("conversation_manager")
        
        history_text = self._format_history_for_prompt(history) if history else "New conversation"
        last_props_text = self._format_properties_brief(last_properties) if last_properties else "None"
        
        prompt = prompt_template.format(
            history=history_text,
            user_query=user_query,
            agent_output=agent_output,
            intent=intent,
            favorites_count=favorites_count,
            last_properties=last_props_text
        )
        
        return self.generate(prompt=prompt)
    
    def _format_properties_for_prompt(self, properties: List[Dict[str, Any]]) -> str:
        """Format properties list for inclusion in prompts."""
        if not properties:
            return "No properties found."
        
        lines = []
        for i, prop in enumerate(properties, 1):
            # Handle both realtor-data schema and standard schema
            prop_id = prop.get('id', 'N/A')
            prop_type = prop.get('property_type') or prop.get('Type') or prop.get('status') or 'N/A'
            
            price = prop.get('price') or prop.get('Price') or 0
            price_str = f"${price:,.0f}" if price else "N/A"
            
            beds = prop.get('bedrooms') or prop.get('Bedrooms') or prop.get('bed') or 'N/A'
            baths = prop.get('bathrooms') or prop.get('Bathrooms') or prop.get('bath') or 'N/A'
            sqft = prop.get('sqft') or prop.get('Size') or prop.get('house_size') or 'N/A'
            sqft_str = f"{float(sqft):,.0f} sqft" if sqft and sqft != 'N/A' else "N/A"
            
            city = prop.get('city') or prop.get('City') or 'N/A'
            state = prop.get('state') or prop.get('State') or 'N/A'
            address = prop.get('address') or prop.get('Address') or prop.get('street') or 'N/A'
            
            # Optional: lot size for realtor data
            acre_lot = prop.get('acre_lot') or prop.get('AcreLot')
            lot_str = f", Lot: {float(acre_lot):.2f} acres" if acre_lot else ""
            
            lines.append(
                f"{i}. ID: {prop_id}\n"
                f"   Status/Type: {prop_type}\n"
                f"   Price: {price_str}\n"
                f"   Beds: {beds}, Baths: {baths}\n"
                f"   Size: {sqft_str}{lot_str}\n"
                f"   Location: {city}, {state}\n"
                f"   Address: {address}"
            )
        
        return "\n".join(lines)
    
    def _format_property_for_prompt(self, prop: Dict[str, Any]) -> str:
        """Format a single property for inclusion in prompts."""
        # Handle both realtor-data schema and standard schema
        prop_type = prop.get('property_type') or prop.get('Type') or prop.get('status') or 'N/A'
        
        price = prop.get('price') or prop.get('Price') or 0
        price_str = f"${price:,.0f}" if price else "N/A"
        
        beds = prop.get('bedrooms') or prop.get('Bedrooms') or prop.get('bed') or 'N/A'
        baths = prop.get('bathrooms') or prop.get('Bathrooms') or prop.get('bath') or 'N/A'
        sqft = prop.get('sqft') or prop.get('Size') or prop.get('house_size') or 'N/A'
        sqft_str = f"{float(sqft):,.0f} sqft" if sqft and sqft != 'N/A' else "N/A"
        
        city = prop.get('city') or prop.get('City') or 'N/A'
        state = prop.get('state') or prop.get('State') or 'N/A'
        address = prop.get('address') or prop.get('Address') or prop.get('street') or 'N/A'
        
        acre_lot = prop.get('acre_lot') or prop.get('AcreLot')
        lot_str = f"\nLot Size: {float(acre_lot):.2f} acres" if acre_lot else ""
        
        return (
            f"Type/Status: {prop_type}\n"
            f"Price: {price_str}\n"
            f"Beds: {beds}, Baths: {baths}\n"
            f"Size: {sqft_str}{lot_str}\n"
            f"Location: {city}, {state}\n"
            f"Address: {address}"
        )
    
    def _format_properties_brief(self, properties: List[Dict[str, Any]]) -> str:
        """Format properties list briefly for context."""
        if not properties:
            return "None"
        
        return ", ".join([
            f"{p.get('address', p.get('Address', p.get('id', 'Unknown')))}"
            for p in properties[:5]
        ])
    
    def _format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompts."""
        if not history:
            return "New conversation (no history)"
        
        lines = []
        for entry in history[-5:]:  # Last 5 exchanges
            role = entry.get("role", "user").capitalize()
            content = entry.get("content", "")[:200]  # Truncate long messages
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _format_stats_for_prompt(self, stats: Dict[str, Any]) -> str:
        """Format statistics for prompts."""
        if not stats:
            return "No statistics available."
        
        lines = []
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"- {key}: ${value:,.2f}" if "price" in key.lower() else f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        try:
            # Simple test with minimal tokens
            response = self._client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get or create the LLM service singleton.
    
    Returns:
        LLMService instance
    """
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMService()
    
    return _llm_service


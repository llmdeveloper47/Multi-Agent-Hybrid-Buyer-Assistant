"""
Superlinked service for semantic search of property listings.

Uses Superlinked framework for multi-modal vector search with:
- Text similarity for property descriptions
- Number spaces for price, bedrooms, bathrooms filtering
- Natural language query support with OpenAI integration
- Qdrant as the persistent vector database backend
"""

import os
from typing import List, Dict, Any, Optional
from functools import lru_cache

from superlinked import framework as sl

from ..config import get_settings


# ============================================================
# Superlinked Schema Definition
# ============================================================

class Property(sl.Schema):
    """
    Superlinked schema for real estate properties.
    
    Maps to realtor-data.csv columns:
    brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date
    """
    id: sl.IdField
    description: sl.String  # Combined text description for semantic search
    price: sl.Float
    bedrooms: sl.Integer
    bathrooms: sl.Float
    house_size: sl.Float  # sqft
    acre_lot: sl.Float
    city: sl.String
    state: sl.String
    status: sl.String


class SuperlinkedService:
    """
    Service for managing Superlinked vector search with Qdrant backend.
    
    Provides multi-modal search combining:
    - Semantic text search on property descriptions
    - Numeric filtering on price, beds, baths, size
    - Location-based filtering
    - Persistent storage via Qdrant vector database
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._property_schema = Property()
        self._spaces = {}
        self._index = None
        self._query = None
        self._source = None
        self._executor = None
        self._app = None
        self._vector_database = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Superlinked spaces, index, query, and Qdrant backend."""
        
        # Define embedding spaces
        
        # Text similarity space for property descriptions
        self._spaces['description'] = sl.TextSimilaritySpace(
            text=self._property_schema.description,
            model="Alibaba-NLP/gte-large-en-v1.5"  # High-quality embedding model
        )
        
        # Number spaces for filtering with different modes
        self._spaces['price'] = sl.NumberSpace(
            number=self._property_schema.price,
            min_value=0,
            max_value=100_000_000,
            mode=sl.Mode.SIMILAR  # Find similar prices
        )
        
        self._spaces['bedrooms'] = sl.NumberSpace(
            number=self._property_schema.bedrooms,
            min_value=0,
            max_value=20,
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['bathrooms'] = sl.NumberSpace(
            number=self._property_schema.bathrooms,
            min_value=0,
            max_value=20,
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['house_size'] = sl.NumberSpace(
            number=self._property_schema.house_size,
            min_value=0,
            max_value=100_000,
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['acre_lot'] = sl.NumberSpace(
            number=self._property_schema.acre_lot,
            min_value=0,
            max_value=1000,
            mode=sl.Mode.SIMILAR
        )
        
        # Create index with all spaces
        self._index = sl.Index(
            [
                self._spaces['description'],
                self._spaces['price'],
                self._spaces['bedrooms'],
                self._spaces['bathrooms'],
                self._spaces['house_size'],
                self._spaces['acre_lot'],
            ],
            fields=[
                self._property_schema.price,
                self._property_schema.bedrooms,
                self._property_schema.bathrooms,
                self._property_schema.house_size,
                self._property_schema.acre_lot,
                self._property_schema.city,
                self._property_schema.state,
                self._property_schema.status,
            ]
        )
        
        # Define query with dynamic weights
        self._query = (
            sl.Query(
                self._index,
                weights={
                    self._spaces['description']: sl.Param("description_weight", default=1.0),
                    self._spaces['price']: sl.Param("price_weight", default=0.3),
                    self._spaces['bedrooms']: sl.Param("bedrooms_weight", default=0.2),
                    self._spaces['bathrooms']: sl.Param("bathrooms_weight", default=0.1),
                    self._spaces['house_size']: sl.Param("house_size_weight", default=0.1),
                    self._spaces['acre_lot']: sl.Param("acre_lot_weight", default=0.05),
                },
            )
            .find(self._property_schema)
            .similar(
                self._spaces['description'],
                sl.Param(
                    "description_query",
                    description="The text describing what kind of property the user is looking for."
                ),
            )
            .similar(
                self._spaces['price'],
                sl.Param("target_price", description="Target price for the property.", default=None),
            )
            .similar(
                self._spaces['bedrooms'],
                sl.Param("target_bedrooms", description="Target number of bedrooms.", default=None),
            )
            .similar(
                self._spaces['bathrooms'],
                sl.Param("target_bathrooms", description="Target number of bathrooms.", default=None),
            )
            .filter(
                self._property_schema.state == sl.Param("state_filter", default=None)
            )
            .filter(
                self._property_schema.city == sl.Param("city_filter", default=None)
            )
            .filter(
                self._property_schema.price <= sl.Param("max_price", default=None)
            )
            .filter(
                self._property_schema.price >= sl.Param("min_price", default=None)
            )
            .filter(
                self._property_schema.bedrooms >= sl.Param("min_bedrooms", default=None)
            )
            .select_all()
            .limit(sl.Param("limit", default=10))
            .with_natural_query(
                sl.Param("natural_language_query"),
                sl.OpenAIClientConfig(
                    api_key=self.settings.OPENAI_API_KEY,
                    model=self.settings.OPENAI_MODEL
                )
            )
        )
        
        # Initialize Qdrant vector database
        self._vector_database = self._create_qdrant_database()
        
        # Initialize source and executor with Qdrant backend
        self._source = sl.InMemorySource(self._property_schema)
        
        if self._vector_database:
            # Use RestExecutor with Qdrant for production
            self._executor = sl.RestExecutor(
                sources=[self._source],
                indices=[self._index],
                queries=[sl.RestQuery(sl.RestDescriptor("property_search"), self._query)],
                vector_database=self._vector_database,
            )
            print(f"Superlinked initialized with Qdrant backend at {self._get_qdrant_url()}")
        else:
            # Fallback to in-memory for development/testing
            self._executor = sl.InMemoryExecutor(
                sources=[self._source],
                indices=[self._index]
            )
            print("Superlinked initialized with in-memory backend (Qdrant not configured)")
        
        self._app = self._executor.run()
    
    def _create_qdrant_database(self) -> Optional[sl.QdrantVectorDatabase]:
        """
        Create Qdrant vector database connection.
        
        Supports:
        - Local Qdrant server (docker or standalone)
        - Qdrant Cloud
        
        Returns:
            QdrantVectorDatabase instance or None if not configured
        """
        try:
            qdrant_url = self._get_qdrant_url()
            
            # Check if using Qdrant Cloud (has API key)
            if self.settings.QDRANT_API_KEY:
                # Qdrant Cloud configuration
                return sl.QdrantVectorDatabase(
                    url=qdrant_url,
                    api_key=self.settings.QDRANT_API_KEY,
                    default_query_limit=self.settings.SEARCH_TOP_K,
                )
            else:
                # Local Qdrant (no API key needed)
                return sl.QdrantVectorDatabase(
                    url=qdrant_url,
                    default_query_limit=self.settings.SEARCH_TOP_K,
                )
        except Exception as e:
            print(f"Warning: Could not initialize Qdrant database: {e}")
            print("Falling back to in-memory storage")
            return None
    
    def _get_qdrant_url(self) -> str:
        """Get Qdrant URL from settings."""
        if hasattr(self.settings, 'QDRANT_URL') and self.settings.QDRANT_URL:
            return self.settings.QDRANT_URL
        
        # Build URL from host and port
        host = self.settings.QDRANT_HOST
        port = self.settings.QDRANT_PORT
        
        # Use http for local, https for cloud
        protocol = "https" if self.settings.QDRANT_API_KEY else "http"
        
        return f"{protocol}://{host}:{port}"
    
    def ingest_properties(self, properties: List[Dict[str, Any]]) -> int:
        """
        Ingest properties into the Superlinked index.
        
        Args:
            properties: List of property dictionaries
            
        Returns:
            Number of properties ingested
        """
        # Transform properties to Superlinked format
        sl_properties = []
        for prop in properties:
            sl_prop = self._transform_to_superlinked(prop)
            if sl_prop:
                sl_properties.append(sl_prop)
        
        # Ingest into source
        self._source.put(sl_properties)
        
        return len(sl_properties)
    
    def _transform_to_superlinked(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a property dict to Superlinked schema format.
        
        Args:
            prop: Raw property dictionary
            
        Returns:
            Transformed property for Superlinked
        """
        try:
            # Create text description for semantic search
            description = self._create_description(prop)
            
            return {
                "id": str(prop.get("id", "")),
                "description": description,
                "price": float(prop.get("price") or prop.get("Price") or 0),
                "bedrooms": int(prop.get("bed") or prop.get("Bedrooms") or prop.get("bedrooms") or 0),
                "bathrooms": float(prop.get("bath") or prop.get("Bathrooms") or prop.get("bathrooms") or 0),
                "house_size": float(prop.get("house_size") or prop.get("Size") or prop.get("sqft") or 0),
                "acre_lot": float(prop.get("acre_lot") or prop.get("AcreLot") or 0),
                "city": str(prop.get("city") or prop.get("City") or ""),
                "state": str(prop.get("state") or prop.get("State") or ""),
                "status": str(prop.get("status") or prop.get("Type") or "for_sale"),
            }
        except Exception as e:
            print(f"Error transforming property: {e}")
            return None
    
    def _create_description(self, prop: Dict[str, Any]) -> str:
        """Create a text description from property fields."""
        parts = []
        
        status = prop.get("status") or prop.get("Type") or "for sale"
        parts.append(f"Property {status.replace('_', ' ')}")
        
        city = prop.get("city") or prop.get("City")
        state = prop.get("state") or prop.get("State")
        if city and state:
            parts.append(f"located in {city}, {state}")
        
        price = prop.get("price") or prop.get("Price")
        if price:
            parts.append(f"priced at ${float(price):,.0f}")
        
        beds = prop.get("bed") or prop.get("Bedrooms") or prop.get("bedrooms")
        baths = prop.get("bath") or prop.get("Bathrooms") or prop.get("bathrooms")
        if beds:
            parts.append(f"with {int(beds)} bedrooms")
        if baths:
            parts.append(f"and {float(baths)} bathrooms")
        
        sqft = prop.get("house_size") or prop.get("Size") or prop.get("sqft")
        if sqft:
            parts.append(f"{float(sqft):,.0f} square feet")
        
        acre_lot = prop.get("acre_lot") or prop.get("AcreLot")
        if acre_lot:
            parts.append(f"on {float(acre_lot):.2f} acres")
        
        return " ".join(parts)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_natural_language: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for properties using Superlinked.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            filters: Optional filters (min_price, max_price, min_bedrooms, state, city)
            use_natural_language: Whether to use LLM for query understanding
            
        Returns:
            List of matching properties
        """
        filters = filters or {}
        
        if use_natural_language:
            # Use Superlinked's natural language query feature
            result = self._app.query(
                self._query,
                natural_language_query=query,
                limit=top_k,
                # Pass explicit filters if provided
                min_price=filters.get("min_price"),
                max_price=filters.get("max_price"),
                min_bedrooms=filters.get("min_bedrooms"),
                state_filter=filters.get("state"),
                city_filter=filters.get("city"),
            )
        else:
            # Direct parameter query
            result = self._app.query(
                self._query,
                description_query=query,
                limit=top_k,
                min_price=filters.get("min_price"),
                max_price=filters.get("max_price"),
                min_bedrooms=filters.get("min_bedrooms"),
                target_price=filters.get("target_price"),
                target_bedrooms=filters.get("target_bedrooms"),
                state_filter=filters.get("state"),
                city_filter=filters.get("city"),
            )
        
        # Convert results to standard format
        return self._convert_results(result)
    
    def _convert_results(self, result) -> List[Dict[str, Any]]:
        """Convert Superlinked results to standard property format."""
        properties = []
        
        try:
            # Convert to pandas for easier handling
            df = sl.PandasConverter.to_pandas(result)
            
            for _, row in df.iterrows():
                properties.append({
                    "id": str(row.get("id", "")),
                    "Type": row.get("status", "for_sale"),
                    "Price": row.get("price"),
                    "Bedrooms": row.get("bedrooms"),
                    "Bathrooms": row.get("bathrooms"),
                    "Size": row.get("house_size"),
                    "AcreLot": row.get("acre_lot"),
                    "City": row.get("city"),
                    "State": row.get("state"),
                    "score": row.get("_score", 0),
                })
        except Exception as e:
            print(f"Error converting results: {e}")
        
        return properties
    
    def get_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a property by its ID.
        
        Args:
            property_id: The property ID
            
        Returns:
            Property dictionary or None
        """
        # Search by exact ID match
        result = self._app.query(
            self._query,
            description_query=f"property id {property_id}",
            limit=100
        )
        
        results = self._convert_results(result)
        for prop in results:
            if str(prop.get("id")) == str(property_id):
                return prop
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data."""
        backend = "qdrant" if self._vector_database else "in_memory"
        
        stats = {
            "status": "running",
            "backend": backend,
            "spaces": list(self._spaces.keys()),
        }
        
        if self._vector_database:
            stats["qdrant_url"] = self._get_qdrant_url()
        
        return stats
    
    def is_connected(self) -> bool:
        """Check if the service is running."""
        if self._app is None:
            return False
        
        # If using Qdrant, verify connection
        if self._vector_database:
            try:
                # Test Qdrant connection
                from qdrant_client import QdrantClient
                client = QdrantClient(
                    url=self._get_qdrant_url(),
                    api_key=self.settings.QDRANT_API_KEY if self.settings.QDRANT_API_KEY else None
                )
                client.get_collections()
                return True
            except Exception:
                return False
        
        return True


# Singleton instance
_superlinked_service: Optional[SuperlinkedService] = None


def get_superlinked_service() -> SuperlinkedService:
    """
    Get or create the Superlinked service singleton.
    
    Returns:
        SuperlinkedService instance
    """
    global _superlinked_service
    
    if _superlinked_service is None:
        _superlinked_service = SuperlinkedService()
    
    return _superlinked_service


"""
Superlinked service for semantic search of property listings.

Uses Superlinked framework for multi-modal vector search with:
- Text similarity for property descriptions
- Number spaces for price, bedrooms, bathrooms filtering
- Natural language query support with OpenAI integration
- Qdrant as the persistent vector database backend

Dataset: realtor-data.csv
Columns: brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from functools import lru_cache
from pathlib import Path

from superlinked import framework as sl

from ..config import get_settings


# ============================================================
# Dataset Statistics (from realtor-data.csv analysis)
# These are used for NumberSpace min/max values
# ============================================================

DATASET_STATS = {
    "price": {"min": 100000.0, "max": 515000000.0},
    "bed": {"min": 1, "max": 50},  # Capped at realistic max (dataset has outliers up to 444)
    "bath": {"min": 1, "max": 30},  # Capped at realistic max (dataset has outliers up to 222)
    "acre_lot": {"min": 0.0, "max": 100000.0},
    "house_size": {"min": 100.0, "max": 1560780.0},
    "zip_code": {"min": 602, "max": 99901},
}


# ============================================================
# Superlinked Schema Definition
# ============================================================

@sl.schema
class Property:
    """
    Superlinked schema for real estate properties.
    
    Maps directly to realtor-data.csv columns:
    - brokered_by: float (broker ID)
    - status: string (for_sale, sold)
    - price: float
    - bed: int (number of bedrooms)
    - bath: float (number of bathrooms)
    - acre_lot: float
    - street: string (street identifier)
    - city: string
    - state: string
    - zip_code: int
    - house_size: float (sqft)
    - prev_sold_date: string
    """
    id: sl.IdField
    description: sl.String  # Generated text for semantic search
    
    # Numeric fields (from CSV)
    price: sl.Float
    bed: sl.Integer  # bedrooms
    bath: sl.Float   # bathrooms
    acre_lot: sl.Float
    house_size: sl.Float  # sqft
    zip_code: sl.Integer
    
    # String/categorical fields (from CSV)
    status: sl.String
    city: sl.String
    state: sl.String
    street: sl.String
    brokered_by: sl.String
    prev_sold_date: sl.String


def load_dataset_stats(data_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Load actual min/max statistics from the dataset.
    
    Args:
        data_path: Path to the CSV file. If None, uses default from settings.
        
    Returns:
        Dictionary with min/max values for each numeric column
    """
    settings = get_settings()
    
    if data_path is None:
        data_path = Path(settings.DATA_PATH) / settings.DATASET_FILENAME
    
    stats = DATASET_STATS.copy()
    
    try:
        # Read dataset to get actual statistics
        df = pd.read_csv(data_path)
        
        numeric_cols = {
            "price": ("price", 100000.0, 515000000.0),
            "bed": ("bed", 1, 50),  # Cap outliers
            "bath": ("bath", 1, 30),  # Cap outliers
            "acre_lot": ("acre_lot", 0.0, 100000.0),
            "house_size": ("house_size", 100.0, 1560780.0),
            "zip_code": ("zip_code", 602, 99901),
        }
        
        for key, (col, default_min, default_max) in numeric_cols.items():
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    actual_min = float(valid.min())
                    actual_max = float(valid.max())
                    
                    # For bed/bath, cap at reasonable values to handle outliers
                    if key == "bed":
                        actual_max = min(actual_max, 50)
                    elif key == "bath":
                        actual_max = min(actual_max, 30)
                    
                    stats[key] = {"min": actual_min, "max": actual_max}
                    
        print(f"Loaded dataset statistics from {data_path}")
        for key, val in stats.items():
            print(f"  {key}: min={val['min']}, max={val['max']}")
            
    except Exception as e:
        print(f"Warning: Could not load dataset stats: {e}")
        print("Using default statistics")
    
    return stats


class SuperlinkedService:
    """
    Service for managing Superlinked vector search with Qdrant backend.
    
    Provides multi-modal search combining:
    - Semantic text search on property descriptions
    - Numeric filtering on price, beds, baths, size
    - Location-based filtering
    - Persistent storage via Qdrant vector database
    
    Schema matches realtor-data.csv exactly:
    brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date
    """
    
    def __init__(self, load_stats_from_data: bool = True):
        """
        Initialize the Superlinked service.
        
        Args:
            load_stats_from_data: If True, load min/max stats from dataset file.
                                  If False, use predefined defaults.
        """
        self.settings = get_settings()
        self._property_schema = Property()
        self._spaces = {}
        self._index = None
        self._query = None
        self._source = None
        self._executor = None
        self._app = None
        self._vector_database = None
        
        # Load dataset statistics
        if load_stats_from_data:
            self._stats = load_dataset_stats()
        else:
            self._stats = DATASET_STATS.copy()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Superlinked spaces, index, query, and Qdrant backend."""
        
        # Define embedding spaces
        
        # Text similarity space for property descriptions
        self._spaces['description'] = sl.TextSimilaritySpace(
            text=self._property_schema.description,
            model="Alibaba-NLP/gte-large-en-v1.5"  # High-quality embedding model
        )
        
        # Number spaces using actual dataset statistics
        self._spaces['price'] = sl.NumberSpace(
            number=self._property_schema.price,
            min_value=self._stats["price"]["min"],
            max_value=self._stats["price"]["max"],
            mode=sl.Mode.SIMILAR  # Find similar prices
        )
        
        self._spaces['bed'] = sl.NumberSpace(
            number=self._property_schema.bed,
            min_value=self._stats["bed"]["min"],
            max_value=self._stats["bed"]["max"],
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['bath'] = sl.NumberSpace(
            number=self._property_schema.bath,
            min_value=self._stats["bath"]["min"],
            max_value=self._stats["bath"]["max"],
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['house_size'] = sl.NumberSpace(
            number=self._property_schema.house_size,
            min_value=self._stats["house_size"]["min"],
            max_value=self._stats["house_size"]["max"],
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['acre_lot'] = sl.NumberSpace(
            number=self._property_schema.acre_lot,
            min_value=self._stats["acre_lot"]["min"],
            max_value=self._stats["acre_lot"]["max"],
            mode=sl.Mode.SIMILAR
        )
        
        self._spaces['zip_code'] = sl.NumberSpace(
            number=self._property_schema.zip_code,
            min_value=self._stats["zip_code"]["min"],
            max_value=self._stats["zip_code"]["max"],
            mode=sl.Mode.SIMILAR
        )
        
        # Create index with all spaces
        self._index = sl.Index(
            spaces=[
                self._spaces['description'],
                self._spaces['price'],
                self._spaces['bed'],
                self._spaces['bath'],
                self._spaces['house_size'],
                self._spaces['acre_lot'],
                self._spaces['zip_code'],
            ],
            fields=[
                self._property_schema.price,
                self._property_schema.bed,
                self._property_schema.bath,
                self._property_schema.house_size,
                self._property_schema.acre_lot,
                self._property_schema.zip_code,
                self._property_schema.city,
                self._property_schema.state,
                self._property_schema.status,
                self._property_schema.street,
                self._property_schema.brokered_by,
                self._property_schema.prev_sold_date,
            ]
        )
        
        # Define query with dynamic weights
        self._query = (
            sl.Query(
                self._index,
                weights={
                    self._spaces['description']: sl.Param("description_weight", default=1.0),
                    self._spaces['price']: sl.Param("price_weight", default=0.3),
                    self._spaces['bed']: sl.Param("bed_weight", default=0.2),
                    self._spaces['bath']: sl.Param("bath_weight", default=0.1),
                    self._spaces['house_size']: sl.Param("house_size_weight", default=0.1),
                    self._spaces['acre_lot']: sl.Param("acre_lot_weight", default=0.05),
                    self._spaces['zip_code']: sl.Param("zip_code_weight", default=0.05),
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
                self._spaces['bed'],
                sl.Param("target_bed", description="Target number of bedrooms.", default=None),
            )
            .similar(
                self._spaces['bath'],
                sl.Param("target_bath", description="Target number of bathrooms.", default=None),
            )
            .similar(
                self._spaces['house_size'],
                sl.Param("target_house_size", description="Target house size in sqft.", default=None),
            )
            .filter(
                self._property_schema.state == sl.Param("state_filter", default=None)
            )
            .filter(
                self._property_schema.city == sl.Param("city_filter", default=None)
            )
            .filter(
                self._property_schema.status == sl.Param("status_filter", default=None)
            )
            .filter(
                self._property_schema.price <= sl.Param("max_price", default=None)
            )
            .filter(
                self._property_schema.price >= sl.Param("min_price", default=None)
            )
            .filter(
                self._property_schema.bed >= sl.Param("min_bed", default=None)
            )
            .filter(
                self._property_schema.bath >= sl.Param("min_bath", default=None)
            )
            .filter(
                self._property_schema.house_size >= sl.Param("min_house_size", default=None)
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
            properties: List of property dictionaries (matching realtor-data.csv schema)
            
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
        
        Expects realtor-data.csv columns:
        brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date
        
        Args:
            prop: Raw property dictionary from CSV
            
        Returns:
            Transformed property for Superlinked
        """
        try:
            # Create text description for semantic search
            description = self._create_description(prop)
            
            # Handle potential NaN/None values with defaults
            def safe_float(val, default=0.0):
                try:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(val, default=0):
                try:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return default
                    return int(float(val))
                except (ValueError, TypeError):
                    return default
            
            def safe_str(val, default=""):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return default
                return str(val)
            
            return {
                "id": str(prop.get("id", prop.get("index", ""))),
                "description": description,
                
                # Numeric fields (exact CSV column names)
                "price": safe_float(prop.get("price")),
                "bed": safe_int(prop.get("bed")),
                "bath": safe_float(prop.get("bath")),
                "acre_lot": safe_float(prop.get("acre_lot")),
                "house_size": safe_float(prop.get("house_size")),
                "zip_code": safe_int(prop.get("zip_code")),
                
                # String fields (exact CSV column names)
                "status": safe_str(prop.get("status"), "for_sale"),
                "city": safe_str(prop.get("city")),
                "state": safe_str(prop.get("state")),
                "street": safe_str(prop.get("street")),
                "brokered_by": safe_str(prop.get("brokered_by")),
                "prev_sold_date": safe_str(prop.get("prev_sold_date")),
            }
        except Exception as e:
            print(f"Error transforming property: {e}")
            return None
    
    def _create_description(self, prop: Dict[str, Any]) -> str:
        """
        Create a text description from property fields for semantic search.
        
        Uses exact column names from realtor-data.csv.
        """
        parts = []
        
        # Status
        status = prop.get("status", "for sale")
        if status:
            parts.append(f"Property {str(status).replace('_', ' ')}")
        
        # Location
        city = prop.get("city")
        state = prop.get("state")
        if city and state and not pd.isna(city) and not pd.isna(state):
            parts.append(f"located in {city}, {state}")
        
        zip_code = prop.get("zip_code")
        if zip_code and not pd.isna(zip_code):
            parts.append(f"ZIP {int(zip_code)}")
        
        # Price
        price = prop.get("price")
        if price and not pd.isna(price):
            parts.append(f"priced at ${float(price):,.0f}")
        
        # Beds and baths
        beds = prop.get("bed")
        baths = prop.get("bath")
        if beds and not pd.isna(beds):
            parts.append(f"with {int(beds)} bedrooms")
        if baths and not pd.isna(baths):
            parts.append(f"and {float(baths):.1f} bathrooms")
        
        # Size
        sqft = prop.get("house_size")
        if sqft and not pd.isna(sqft):
            parts.append(f"{float(sqft):,.0f} square feet")
        
        # Lot size
        acre_lot = prop.get("acre_lot")
        if acre_lot and not pd.isna(acre_lot) and float(acre_lot) > 0:
            parts.append(f"on {float(acre_lot):.2f} acres")
        
        # Previous sale
        prev_sold = prop.get("prev_sold_date")
        if prev_sold and not pd.isna(prev_sold):
            parts.append(f"previously sold {prev_sold}")
        
        return " ".join(parts) if parts else "Real estate property"
    
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
            filters: Optional filters matching CSV columns:
                - min_price, max_price
                - min_bed, min_bath
                - min_house_size
                - state, city, status
                - target_price, target_bed, target_bath, target_house_size
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
                min_bed=filters.get("min_bed") or filters.get("min_bedrooms"),
                min_bath=filters.get("min_bath") or filters.get("min_bathrooms"),
                min_house_size=filters.get("min_house_size"),
                state_filter=filters.get("state"),
                city_filter=filters.get("city"),
                status_filter=filters.get("status"),
            )
        else:
            # Direct parameter query
            result = self._app.query(
                self._query,
                description_query=query,
                limit=top_k,
                min_price=filters.get("min_price"),
                max_price=filters.get("max_price"),
                min_bed=filters.get("min_bed") or filters.get("min_bedrooms"),
                min_bath=filters.get("min_bath") or filters.get("min_bathrooms"),
                min_house_size=filters.get("min_house_size"),
                target_price=filters.get("target_price"),
                target_bed=filters.get("target_bed") or filters.get("target_bedrooms"),
                target_bath=filters.get("target_bath") or filters.get("target_bathrooms"),
                target_house_size=filters.get("target_house_size"),
                state_filter=filters.get("state"),
                city_filter=filters.get("city"),
                status_filter=filters.get("status"),
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
                # Return data using exact CSV column names
                properties.append({
                    "id": str(row.get("id", "")),
                    "status": row.get("status", "for_sale"),
                    "price": row.get("price"),
                    "bed": row.get("bed"),
                    "bath": row.get("bath"),
                    "house_size": row.get("house_size"),
                    "acre_lot": row.get("acre_lot"),
                    "city": row.get("city"),
                    "state": row.get("state"),
                    "zip_code": row.get("zip_code"),
                    "street": row.get("street"),
                    "brokered_by": row.get("brokered_by"),
                    "prev_sold_date": row.get("prev_sold_date"),
                    "score": row.get("_score", 0),
                    # Also include aliases for backward compatibility
                    "Price": row.get("price"),
                    "Bedrooms": row.get("bed"),
                    "Bathrooms": row.get("bath"),
                    "Size": row.get("house_size"),
                    "AcreLot": row.get("acre_lot"),
                    "City": row.get("city"),
                    "State": row.get("state"),
                    "Type": row.get("status", "for_sale"),
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
            "dataset_stats": self._stats,
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


def get_superlinked_service(load_stats_from_data: bool = True) -> SuperlinkedService:
    """
    Get or create the Superlinked service singleton.
    
    Args:
        load_stats_from_data: If True, load min/max stats from dataset file on first init.
    
    Returns:
        SuperlinkedService instance
    """
    global _superlinked_service
    
    if _superlinked_service is None:
        _superlinked_service = SuperlinkedService(load_stats_from_data=load_stats_from_data)
    
    return _superlinked_service

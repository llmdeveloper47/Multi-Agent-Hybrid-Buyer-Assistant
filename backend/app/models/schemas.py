"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PropertySummary(BaseModel):
    """Summary of a real estate property."""
    
    id: str = Field(..., description="Unique identifier for the property")
    property_type: Optional[str] = Field(None, description="Type/status of property (e.g., for_sale, Single Family)")
    price: Optional[float] = Field(None, description="Listing price in USD")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    sqft: Optional[float] = Field(None, description="Square footage (house_size)")
    acre_lot: Optional[float] = Field(None, description="Lot size in acres")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State name or abbreviation")
    address: Optional[str] = Field(None, description="Street address")
    zip_code: Optional[str] = Field(None, description="ZIP code")
    description: Optional[str] = Field(None, description="Property description or summary")
    brokered_by: Optional[str] = Field(None, description="Broker ID")
    prev_sold_date: Optional[str] = Field(None, description="Previous sold date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "prop_12345",
                "property_type": "for_sale",
                "price": 450000,
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft": 1800,
                "acre_lot": 0.25,
                "city": "Austin",
                "state": "Texas",
                "address": "123 Maple St",
                "zip_code": "78701"
            }
        }


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""
    
    role: str = Field(..., description="Role of the message sender: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="When the message was sent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Find me 3-bedroom houses in Seattle under $500k",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class QueryRequest(BaseModel):
    """Request model for the main query endpoint."""
    
    session_id: str = Field(..., description="Unique session identifier for the user")
    message: str = Field(..., description="User's query or message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "message": "Show me houses in Austin with 3 bedrooms under $500k"
            }
        }


class QueryResponse(BaseModel):
    """Response model for the main query endpoint."""
    
    answer: str = Field(..., description="Assistant's response to the query")
    intent: Optional[str] = Field(None, description="Detected intent of the query")
    properties: Optional[List[PropertySummary]] = Field(None, description="List of properties if search was performed")
    favorites: Optional[List[PropertySummary]] = Field(None, description="Updated favorites list")
    session_id: str = Field(..., description="Session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "I found 3 houses in Austin that match your criteria...",
                "intent": "search_or_recommend",
                "properties": [],
                "favorites": [],
                "session_id": "sess_abc123"
            }
        }


class FavoriteRequest(BaseModel):
    """Request model for managing favorites."""
    
    session_id: str = Field(..., description="Unique session identifier")
    property_id: str = Field(..., description="Property ID to add/remove from favorites")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "property_id": "prop_12345"
            }
        }


class FavoriteResponse(BaseModel):
    """Response model for favorites operations."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    favorites: List[PropertySummary] = Field(default_factory=list, description="Updated favorites list")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Property added to favorites",
                "favorites": []
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    vector_store_connected: bool = Field(..., description="Whether vector store is connected")
    llm_available: bool = Field(..., description="Whether LLM service is available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "vector_store_connected": True,
                "llm_available": True
            }
        }


class MarketStats(BaseModel):
    """Statistics for market insights."""
    
    area: str = Field(..., description="Geographic area for stats")
    median_price: Optional[float] = Field(None, description="Median listing price")
    average_price: Optional[float] = Field(None, description="Average listing price")
    price_per_sqft: Optional[float] = Field(None, description="Average price per square foot")
    total_listings: Optional[int] = Field(None, description="Number of listings")
    avg_bedrooms: Optional[float] = Field(None, description="Average number of bedrooms")


class ValuationResult(BaseModel):
    """Result of a property valuation."""
    
    estimated_value_low: Optional[float] = Field(None, description="Low end of value estimate")
    estimated_value_high: Optional[float] = Field(None, description="High end of value estimate")
    comparables_count: int = Field(0, description="Number of comparable properties used")
    avg_price_per_sqft: Optional[float] = Field(None, description="Average price per sqft of comparables")
    analysis: str = Field("", description="Textual analysis of the valuation")


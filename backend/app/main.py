"""
FastAPI main application for the Hybrid Buyer Advisor.

This is the entry point for the backend API server.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import logging

from .config import get_settings
from .models.schemas import (
    QueryRequest,
    QueryResponse,
    FavoriteRequest,
    FavoriteResponse,
    HealthResponse,
    PropertySummary,
)
from .services import (
    get_session_service,
    get_superlinked_service,
    get_llm_service,
)
from .workflow import get_workflow
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown events.
    """
    # Startup
    logger.info("Starting Buyer Advisor API...")
    logger.info(f"Version: {__version__}")
    
    # Initialize services (lazy loading will happen on first request)
    try:
        # Verify Superlinked + Qdrant connection
        superlinked = get_superlinked_service()
        stats = superlinked.get_stats()
        logger.info(f"Superlinked initialized. Stats: {stats}")
    except Exception as e:
        logger.warning(f"Superlinked initialization deferred: {e}")
    
    logger.info("Buyer Advisor API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Buyer Advisor API...")


# Create FastAPI application
app = FastAPI(
    title="Hybrid Buyer Advisor API",
    description="""
    A multi-agent AI assistant for real estate buyers.
    
    Features:
    - Natural language property search
    - Personalized recommendations
    - Price valuation estimates
    - Trade-off analysis and comparisons
    - Favorites management
    """,
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Hybrid Buyer Advisor API",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    superlinked_connected = False
    llm_available = False
    
    try:
        superlinked = get_superlinked_service()
        superlinked_connected = superlinked.is_connected()
    except Exception:
        pass
    
    try:
        llm = get_llm_service()
        llm_available = llm.is_available()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if superlinked_connected else "degraded",
        version=__version__,
        vector_store_connected=superlinked_connected,
        llm_available=llm_available,
    )


@app.post("/query", response_model=QueryResponse, tags=["Chat"])
async def query_assistant(request: QueryRequest):
    """
    Main endpoint for querying the assistant.
    
    Send a natural language message and receive a response from the
    multi-agent system.
    """
    logger.info(f"Query from session {request.session_id}: {request.message[:50]}...")
    
    # Get or create session
    session_service = get_session_service()
    session = session_service.get_or_create_session(request.session_id)
    
    try:
        # Get workflow instance
        workflow = get_workflow()
        
        # Run the workflow
        result = workflow.run(
            user_query=request.message,
            session_id=request.session_id,
            favorites=session.favorites,
            history=session.history,
            last_shown_properties=session.last_shown_properties,
        )
        
        # Extract response
        answer = result.get("answer", "I'm sorry, I couldn't process your request.")
        intent = result.get("intent", "unknown")
        results = result.get("results", [])
        favorites = result.get("favorites", session.favorites)
        
        # Update session
        session.favorites = favorites
        session.add_to_history("user", request.message)
        session.add_to_history("assistant", answer)
        
        if results:
            session.set_last_shown_properties(results)
        
        # Convert results to PropertySummary objects (supports both realtor-data and standard schemas)
        property_summaries = [
            PropertySummary(
                id=str(p.get("id", "")),
                property_type=p.get("property_type") or p.get("Type") or p.get("status"),
                price=p.get("price") or p.get("Price"),
                bedrooms=int(p.get("bedrooms") or p.get("Bedrooms") or p.get("bed") or 0) or None,
                bathrooms=float(p.get("bathrooms") or p.get("Bathrooms") or p.get("bath") or 0) or None,
                sqft=float(p.get("sqft") or p.get("Size") or p.get("house_size") or 0) or None,
                acre_lot=float(p.get("acre_lot") or p.get("AcreLot") or 0) or None,
                city=p.get("city") or p.get("City"),
                state=p.get("state") or p.get("State"),
                address=str(p.get("address") or p.get("Address") or p.get("street") or ""),
                zip_code=str(p.get("zip_code") or p.get("Zip") or ""),
                brokered_by=str(p.get("brokered_by") or p.get("BrokeredBy") or "") or None,
                prev_sold_date=str(p.get("prev_sold_date") or p.get("PrevSoldDate") or "") or None,
            )
            for p in results[:10]
        ]
        
        favorite_summaries = [
            PropertySummary(
                id=str(f.get("id", "")),
                property_type=f.get("property_type") or f.get("Type") or f.get("status"),
                price=f.get("price") or f.get("Price"),
                bedrooms=int(f.get("bedrooms") or f.get("Bedrooms") or f.get("bed") or 0) or None,
                bathrooms=float(f.get("bathrooms") or f.get("Bathrooms") or f.get("bath") or 0) or None,
                sqft=float(f.get("sqft") or f.get("Size") or f.get("house_size") or 0) or None,
                acre_lot=float(f.get("acre_lot") or f.get("AcreLot") or 0) or None,
                city=f.get("city") or f.get("City"),
                state=f.get("state") or f.get("State"),
                address=str(f.get("address") or f.get("Address") or f.get("street") or ""),
                zip_code=str(f.get("zip_code") or f.get("Zip") or ""),
                brokered_by=str(f.get("brokered_by") or f.get("BrokeredBy") or "") or None,
                prev_sold_date=str(f.get("prev_sold_date") or f.get("PrevSoldDate") or "") or None,
            )
            for f in favorites
        ]
        
        logger.info(f"Response generated for session {request.session_id}, intent: {intent}")
        
        return QueryResponse(
            answer=answer,
            intent=intent,
            properties=property_summaries if property_summaries else None,
            favorites=favorite_summaries if favorite_summaries else None,
            session_id=request.session_id,
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/favorites/{session_id}", response_model=FavoriteResponse, tags=["Favorites"])
async def get_favorites(session_id: str):
    """
    Get the current favorites list for a session.
    """
    session_service = get_session_service()
    session = session_service.get_session(session_id)
    
    if not session:
        return FavoriteResponse(
            success=True,
            message="No session found",
            favorites=[],
        )
    
    favorites = [
        PropertySummary(
            id=str(f.get("id", "")),
            property_type=f.get("property_type") or f.get("Type") or f.get("status"),
            price=f.get("price") or f.get("Price"),
            bedrooms=int(f.get("bedrooms") or f.get("Bedrooms") or f.get("bed") or 0) or None,
            bathrooms=float(f.get("bathrooms") or f.get("Bathrooms") or f.get("bath") or 0) or None,
            sqft=float(f.get("sqft") or f.get("Size") or f.get("house_size") or 0) or None,
            acre_lot=float(f.get("acre_lot") or f.get("AcreLot") or 0) or None,
            city=f.get("city") or f.get("City"),
            state=f.get("state") or f.get("State"),
            address=str(f.get("address") or f.get("Address") or f.get("street") or ""),
            zip_code=str(f.get("zip_code") or f.get("Zip") or ""),
        )
        for f in session.favorites
    ]
    
    return FavoriteResponse(
        success=True,
        message=f"Found {len(favorites)} favorites",
        favorites=favorites,
    )


@app.post("/favorites", response_model=FavoriteResponse, tags=["Favorites"])
async def add_favorite(request: FavoriteRequest):
    """
    Add a property to favorites.
    """
    session_service = get_session_service()
    
    # Get property details from Superlinked
    try:
        superlinked = get_superlinked_service()
        property_data = superlinked.get_by_id(request.property_id)
        
        if not property_data:
            raise HTTPException(
                status_code=404,
                detail=f"Property {request.property_id} not found"
            )
        
        # Add to session favorites
        added = session_service.add_favorite(request.session_id, property_data)
        
        session = session_service.get_session(request.session_id)
        favorites = [
            PropertySummary(
                id=str(f.get("id", "")),
                property_type=f.get("property_type") or f.get("Type") or f.get("status"),
                price=f.get("price") or f.get("Price"),
                bedrooms=int(f.get("bedrooms") or f.get("Bedrooms") or f.get("bed") or 0) or None,
                bathrooms=float(f.get("bathrooms") or f.get("Bathrooms") or f.get("bath") or 0) or None,
                sqft=float(f.get("sqft") or f.get("Size") or f.get("house_size") or 0) or None,
                acre_lot=float(f.get("acre_lot") or f.get("AcreLot") or 0) or None,
                city=f.get("city") or f.get("City"),
                state=f.get("state") or f.get("State"),
                address=str(f.get("address") or f.get("Address") or f.get("street") or ""),
                zip_code=str(f.get("zip_code") or f.get("Zip") or ""),
            )
            for f in session.favorites
        ]
        
        return FavoriteResponse(
            success=True,
            message="Property added to favorites" if added else "Property already in favorites",
            favorites=favorites,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding favorite: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add favorite: {str(e)}"
        )


@app.delete("/favorites", response_model=FavoriteResponse, tags=["Favorites"])
async def remove_favorite(session_id: str, property_id: str):
    """
    Remove a property from favorites.
    """
    session_service = get_session_service()
    
    removed = session_service.remove_favorite(session_id, property_id)
    session = session_service.get_session(session_id)
    
    favorites = []
    if session:
        favorites = [
            PropertySummary(
                id=str(f.get("id", "")),
                property_type=f.get("property_type") or f.get("Type") or f.get("status"),
                price=f.get("price") or f.get("Price"),
                bedrooms=int(f.get("bedrooms") or f.get("Bedrooms") or f.get("bed") or 0) or None,
                bathrooms=float(f.get("bathrooms") or f.get("Bathrooms") or f.get("bath") or 0) or None,
                sqft=float(f.get("sqft") or f.get("Size") or f.get("house_size") or 0) or None,
                acre_lot=float(f.get("acre_lot") or f.get("AcreLot") or 0) or None,
                city=f.get("city") or f.get("City"),
                state=f.get("state") or f.get("State"),
                address=str(f.get("address") or f.get("Address") or f.get("street") or ""),
                zip_code=str(f.get("zip_code") or f.get("Zip") or ""),
            )
            for f in session.favorites
        ]
    
    return FavoriteResponse(
        success=removed,
        message="Property removed from favorites" if removed else "Property not found in favorites",
        favorites=favorites,
    )


@app.get("/history/{session_id}", tags=["Session"])
async def get_history(session_id: str, limit: int = 20):
    """
    Get conversation history for a session.
    """
    session_service = get_session_service()
    history = session_service.get_history(session_id, limit)
    
    return {
        "session_id": session_id,
        "history": history,
        "count": len(history),
    }


@app.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """
    Delete a session and all its data.
    """
    session_service = get_session_service()
    deleted = session_service.delete_session(session_id)
    
    return {
        "success": deleted,
        "message": "Session deleted" if deleted else "Session not found",
    }


@app.get("/workflow/diagram", tags=["Debug"])
async def get_workflow_diagram():
    """
    Get a visual representation of the workflow graph.
    """
    workflow = get_workflow()
    return {
        "diagram": workflow.get_graph_visualization(),
    }


@app.get("/stats", tags=["Debug"])
async def get_stats():
    """
    Get system statistics (for debugging/monitoring).
    """
    session_service = get_session_service()
    
    stats = {
        "active_sessions": session_service.get_active_session_count(),
    }
    
    try:
        superlinked = get_superlinked_service()
        stats["superlinked"] = superlinked.get_stats()
    except Exception as e:
        stats["superlinked"] = {"error": str(e)}
    
    return stats


# ============================================================
# Run with Uvicorn (for development)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


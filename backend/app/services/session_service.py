"""
Session management service for maintaining user state across requests.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, field


@dataclass
class SessionData:
    """Data structure for a user session."""
    
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    favorites: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    last_shown_properties: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """Update last accessed time."""
        self.last_accessed = datetime.now()
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep history manageable
        if len(self.history) > 100:
            self.history = self.history[-50:]
    
    def add_favorite(self, property_data: Dict[str, Any]) -> bool:
        """
        Add a property to favorites.
        
        Args:
            property_data: Property dictionary to add
            
        Returns:
            True if added, False if already exists
        """
        prop_id = property_data.get("id")
        if prop_id and not any(f.get("id") == prop_id for f in self.favorites):
            self.favorites.append(property_data)
            return True
        return False
    
    def remove_favorite(self, property_id: str) -> bool:
        """
        Remove a property from favorites.
        
        Args:
            property_id: ID of property to remove
            
        Returns:
            True if removed, False if not found
        """
        original_len = len(self.favorites)
        self.favorites = [f for f in self.favorites if f.get("id") != property_id]
        return len(self.favorites) < original_len
    
    def get_favorite_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get a favorite property by ID."""
        for fav in self.favorites:
            if fav.get("id") == property_id:
                return fav
        return None
    
    def set_last_shown_properties(self, properties: List[Dict[str, Any]]):
        """Update the last shown properties."""
        self.last_shown_properties = properties[:10]  # Keep last 10
    
    def get_property_by_reference(self, reference: str) -> Optional[Dict[str, Any]]:
        """
        Get a property by reference (index, address, or ID).
        
        Args:
            reference: Could be "first", "second", "1", "2", an address, or ID
            
        Returns:
            Property dict if found, None otherwise
        """
        reference_lower = reference.lower().strip()
        
        # Check ordinal references
        ordinal_map = {
            "first": 0, "1": 0, "1st": 0,
            "second": 1, "2": 1, "2nd": 1,
            "third": 2, "3": 2, "3rd": 2,
            "fourth": 3, "4": 3, "4th": 3,
            "fifth": 4, "5": 4, "5th": 4,
        }
        
        if reference_lower in ordinal_map:
            idx = ordinal_map[reference_lower]
            if idx < len(self.last_shown_properties):
                return self.last_shown_properties[idx]
        
        # Check by ID
        for prop in self.last_shown_properties:
            if prop.get("id") == reference:
                return prop
        
        # Check by address (partial match)
        for prop in self.last_shown_properties:
            address = prop.get("address", prop.get("Address", "")).lower()
            if reference_lower in address:
                return prop
        
        # Check favorites as well
        for fav in self.favorites:
            if fav.get("id") == reference:
                return fav
            address = fav.get("address", fav.get("Address", "")).lower()
            if reference_lower in address:
                return fav
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "favorites": self.favorites,
            "history": self.history,
            "last_shown_properties": self.last_shown_properties,
            "user_preferences": self.user_preferences
        }


class SessionService:
    """
    Service for managing user sessions.
    
    Maintains session data in memory with optional cleanup of stale sessions.
    For production, this should be replaced with Redis or a database.
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._session_timeout = timedelta(hours=session_timeout_hours)
    
    def get_or_create_session(self, session_id: str) -> SessionData:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData object
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(session_id=session_id)
            
            session = self._sessions[session_id]
            session.touch()
            return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get existing session if it exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData or None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def cleanup_stale_sessions(self) -> int:
        """
        Remove sessions that have been inactive for too long.
        
        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        removed = 0
        
        with self._lock:
            stale_ids = [
                sid for sid, session in self._sessions.items()
                if now - session.last_accessed > self._session_timeout
            ]
            
            for sid in stale_ids:
                del self._sessions[sid]
                removed += 1
        
        return removed
    
    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def update_favorites(
        self,
        session_id: str,
        favorites: List[Dict[str, Any]]
    ) -> SessionData:
        """
        Update the favorites list for a session.
        
        Args:
            session_id: Session identifier
            favorites: New favorites list
            
        Returns:
            Updated SessionData
        """
        session = self.get_or_create_session(session_id)
        session.favorites = favorites
        return session
    
    def add_favorite(
        self,
        session_id: str,
        property_data: Dict[str, Any]
    ) -> bool:
        """
        Add a property to session favorites.
        
        Args:
            session_id: Session identifier
            property_data: Property to add
            
        Returns:
            True if added successfully
        """
        session = self.get_or_create_session(session_id)
        return session.add_favorite(property_data)
    
    def remove_favorite(self, session_id: str, property_id: str) -> bool:
        """
        Remove a property from session favorites.
        
        Args:
            session_id: Session identifier
            property_id: Property ID to remove
            
        Returns:
            True if removed successfully
        """
        session = self.get_or_create_session(session_id)
        return session.remove_favorite(property_id)
    
    def add_to_history(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str
    ):
        """
        Add a conversation exchange to session history.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
        """
        session = self.get_or_create_session(session_id)
        session.add_to_history("user", user_message)
        session.add_to_history("assistant", assistant_response)
    
    def get_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        session = self.get_session(session_id)
        if session:
            return session.history[-limit:]
        return []
    
    def set_last_shown_properties(
        self,
        session_id: str,
        properties: List[Dict[str, Any]]
    ):
        """
        Update the last shown properties for a session.
        
        Args:
            session_id: Session identifier
            properties: List of properties shown
        """
        session = self.get_or_create_session(session_id)
        session.set_last_shown_properties(properties)


# Singleton instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """
    Get or create the session service singleton.
    
    Returns:
        SessionService instance
    """
    global _session_service
    
    if _session_service is None:
        _session_service = SessionService()
    
    return _session_service


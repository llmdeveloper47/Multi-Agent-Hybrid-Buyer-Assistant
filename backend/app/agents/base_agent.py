"""
Base agent class providing common functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..config import load_system_prompt, get_settings
from ..models.state import BuyerAdvisorState
from ..services import get_llm_service, get_superlinked_service


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality like prompt loading and service access.
    Uses Superlinked for vector search when enabled.
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the agent.
        
        Args:
            agent_name: Name of the agent (used for loading prompts)
        """
        self.agent_name = agent_name
        self._system_prompt: Optional[str] = None
        self._settings = get_settings()
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        if self._system_prompt is None:
            self._system_prompt = load_system_prompt(self.agent_name)
        return self._system_prompt
    
    def reload_prompt(self):
        """Force reload of the system prompt from file."""
        self._system_prompt = load_system_prompt(self.agent_name)
    
    @property
    def llm(self):
        """Get the LLM service."""
        return get_llm_service()
    
    @property
    def vector_store(self):
        """
        Get the vector store service (Superlinked with Qdrant backend).
        """
        return get_superlinked_service()
    
    @property
    def superlinked(self):
        """Get the Superlinked service directly."""
        return get_superlinked_service()
    
    @abstractmethod
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    def __call__(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """Allow agents to be called directly."""
        return self.process(state)


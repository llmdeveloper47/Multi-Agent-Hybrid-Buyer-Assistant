"""
Conversation Manager - Orchestrates dialogue and composes final responses.
"""

from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..models.state import BuyerAdvisorState


class ConversationManager(BaseAgent):
    """
    Conversation Manager for orchestrating dialogue and composing responses.
    
    Ensures conversational continuity, maintains context, and formats final output.
    """
    
    def __init__(self):
        super().__init__("conversation_manager")
    
    def process(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Process the agent output and compose the final response.
        
        Args:
            state: Current workflow state (after specialist agent processing)
            
        Returns:
            Updated state with final answer
        """
        agent_output = state.get("agent_output", "")
        user_query = state.get("user_query", "")
        intent = state.get("intent", "")
        history = state.get("history", [])
        favorites = state.get("favorites", [])
        results = state.get("results", [])
        
        # If there was an error, handle gracefully
        if state.get("error"):
            state["answer"] = self._handle_error(state)
            return state
        
        # Check if agent already produced a good response
        if agent_output and len(agent_output) > 50:
            # Agent output is substantial, use it (possibly with light post-processing)
            final_response = self._post_process_response(
                agent_output, intent, state
            )
        else:
            # Need to compose a response using LLM
            try:
                final_response = self.llm.compose_final_response(
                    agent_output=agent_output,
                    user_query=user_query,
                    intent=intent,
                    history=history,
                    favorites_count=len(favorites),
                    last_properties=results[:5] if results else None
                )
            except Exception as e:
                final_response = agent_output or (
                    "I apologize, but I encountered an issue processing your request. "
                    "Could you please try rephrasing your question?"
                )
        
        state["answer"] = final_response
        
        # Update last shown properties if we have results
        if results:
            state["last_shown_properties"] = results[:10]
        
        return state
    
    def _post_process_response(
        self,
        response: str,
        intent: str,
        state: BuyerAdvisorState
    ) -> str:
        """
        Post-process the agent response for consistency.
        
        Args:
            response: Raw agent response
            intent: Detected intent
            state: Current state
            
        Returns:
            Processed response
        """
        # Add contextual follow-up suggestions based on intent
        follow_up = self._get_follow_up_suggestion(intent, state)
        
        if follow_up and not response.endswith(follow_up):
            response = response.rstrip() + "\n\n" + follow_up
        
        return response
    
    def _get_follow_up_suggestion(
        self,
        intent: str,
        state: BuyerAdvisorState
    ) -> Optional[str]:
        """
        Get contextual follow-up suggestions.
        
        Args:
            intent: Detected intent
            state: Current state
            
        Returns:
            Follow-up suggestion or None
        """
        results = state.get("results", [])
        favorites = state.get("favorites", [])
        
        if intent == "search_or_recommend" and results:
            suggestions = ["💡 Tips:"]
            if not any(r.get("id") in [f.get("id") for f in favorites] for r in results[:3]):
                suggestions.append("Say 'add to favorites' to save a property")
            if len(results) >= 2:
                suggestions.append("'Compare the first two' to see a detailed comparison")
            suggestions.append("'Tell me more about the first one' for details")
            
            if len(suggestions) > 1:
                return " | ".join(suggestions)
        
        elif intent == "valuation":
            return "💡 Want me to find similar properties? Just ask!"
        
        elif intent == "comparison" and len(favorites) > 2:
            return "💡 You have more favorites to compare. Say 'compare all favorites' to see them all."
        
        return None
    
    def _handle_error(self, state: BuyerAdvisorState) -> str:
        """
        Handle error cases gracefully.
        
        Args:
            state: State with error information
            
        Returns:
            User-friendly error message
        """
        error = state.get("error", "Unknown error")
        error_agent = state.get("error_agent", "system")
        
        # Log the error (in production, this would go to proper logging)
        print(f"Error in {error_agent}: {error}")
        
        # Provide user-friendly message based on error type
        if "search" in error.lower() or "vector" in error.lower():
            return (
                "I'm having trouble searching the property database right now. "
                "Please try again in a moment, or try a different search."
            )
        
        if "llm" in error.lower() or "openai" in error.lower():
            return (
                "I'm experiencing some difficulty generating a response. "
                "Please try again or simplify your question."
            )
        
        # Generic error message
        return (
            "I apologize, but something went wrong while processing your request. "
            "Could you please try again or rephrase your question?"
        )
    
    def add_to_history(
        self,
        state: BuyerAdvisorState,
        user_message: str,
        assistant_response: str
    ) -> BuyerAdvisorState:
        """
        Add an exchange to conversation history.
        
        Args:
            state: Current state
            user_message: User's message
            assistant_response: Assistant's response
            
        Returns:
            Updated state with new history
        """
        history = state.get("history", [])
        
        history.append({
            "role": "user",
            "content": user_message
        })
        history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # Keep history manageable (last 20 messages)
        if len(history) > 20:
            history = history[-20:]
        
        state["history"] = history
        return state


# Singleton instance
_conversation_manager: ConversationManager = None


def get_conversation_manager() -> ConversationManager:
    """Get or create the conversation manager singleton."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


def conversation_manager_node(state: BuyerAdvisorState) -> BuyerAdvisorState:
    """
    LangGraph node function for the Conversation Manager.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final answer
    """
    manager = get_conversation_manager()
    return manager.process(state)


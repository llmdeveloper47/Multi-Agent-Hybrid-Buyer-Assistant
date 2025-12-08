"""
LangGraph workflow definition for the Buyer Advisor multi-agent system.
"""

from typing import Dict, Any, Optional, Callable
from functools import lru_cache

from langgraph.graph import StateGraph, END

from ..models.state import BuyerAdvisorState, IntentType, create_initial_state
from ..agents import (
    router_agent_node,
    filter_agent_node,
    valuation_agent_node,
    comparison_agent_node,
    market_insights_agent_node,
    favorites_agent_node,
    conversation_manager_node,
)


class BuyerAdvisorWorkflow:
    """
    Main workflow class that orchestrates the multi-agent system.
    
    Uses LangGraph to define a directed graph of agent nodes with
    conditional routing based on user intent.
    """
    
    def __init__(self):
        self._graph = None
        self._compiled_app = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow graph."""
        # Create the state graph
        self._graph = StateGraph(BuyerAdvisorState)
        
        # Add nodes for each agent
        self._graph.add_node("router", router_agent_node)
        self._graph.add_node("filter", filter_agent_node)
        self._graph.add_node("valuation", valuation_agent_node)
        self._graph.add_node("comparison", comparison_agent_node)
        self._graph.add_node("market_insights", market_insights_agent_node)
        self._graph.add_node("favorites", favorites_agent_node)
        self._graph.add_node("conversation_manager", conversation_manager_node)
        
        # Set the entry point
        self._graph.set_entry_point("router")
        
        # Add conditional edges from router to specialist agents
        self._graph.add_conditional_edges(
            "router",
            self._route_by_intent,
            {
                "filter": "filter",
                "valuation": "valuation",
                "comparison": "comparison",
                "market_insights": "market_insights",
                "favorites": "favorites",
            }
        )
        
        # All specialist agents lead to conversation manager
        for agent in ["filter", "valuation", "comparison", "market_insights", "favorites"]:
            self._graph.add_edge(agent, "conversation_manager")
        
        # Conversation manager is the final node
        self._graph.add_edge("conversation_manager", END)
        
        # Compile the graph
        self._compiled_app = self._graph.compile()
    
    def _route_by_intent(self, state: BuyerAdvisorState) -> str:
        """
        Route to appropriate agent based on classified intent.
        
        Args:
            state: Current workflow state with intent field populated
            
        Returns:
            Name of the next node to execute
        """
        intent = state.get("intent", "").lower()
        
        intent_to_agent = {
            "search_or_recommend": "filter",
            "search": "filter",
            "recommend": "filter",
            "valuation": "valuation",
            "price": "valuation",
            "comparison": "comparison",
            "compare": "comparison",
            "market_insight": "market_insights",
            "market": "market_insights",
            "favorites": "favorites",
            "favorite": "favorites",
        }
        
        return intent_to_agent.get(intent, "filter")  # Default to filter
    
    def run(
        self,
        user_query: str,
        session_id: str,
        favorites: Optional[list] = None,
        history: Optional[list] = None,
        last_shown_properties: Optional[list] = None
    ) -> BuyerAdvisorState:
        """
        Run the workflow for a user query.
        
        Args:
            user_query: The user's query text
            session_id: Session identifier
            favorites: Current favorites list
            history: Conversation history
            last_shown_properties: Properties from last response
            
        Returns:
            Final state after workflow execution
        """
        # Create initial state
        initial_state = create_initial_state(
            user_query=user_query,
            session_id=session_id,
            favorites=favorites or [],
            history=history or [],
            last_shown_properties=last_shown_properties or []
        )
        
        # Run the workflow
        result = self._compiled_app.invoke(initial_state)
        
        return result
    
    def run_with_state(self, state: BuyerAdvisorState) -> BuyerAdvisorState:
        """
        Run the workflow with a pre-built state.
        
        Args:
            state: Pre-configured state
            
        Returns:
            Final state after workflow execution
        """
        return self._compiled_app.invoke(state)
    
    async def arun(
        self,
        user_query: str,
        session_id: str,
        favorites: Optional[list] = None,
        history: Optional[list] = None,
        last_shown_properties: Optional[list] = None
    ) -> BuyerAdvisorState:
        """
        Async version of run.
        
        Args:
            user_query: The user's query text
            session_id: Session identifier
            favorites: Current favorites list
            history: Conversation history
            last_shown_properties: Properties from last response
            
        Returns:
            Final state after workflow execution
        """
        initial_state = create_initial_state(
            user_query=user_query,
            session_id=session_id,
            favorites=favorites or [],
            history=history or [],
            last_shown_properties=last_shown_properties or []
        )
        
        result = await self._compiled_app.ainvoke(initial_state)
        
        return result
    
    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the workflow graph.
        
        Returns:
            ASCII diagram of the graph
        """
        diagram = """
        ┌─────────────────────────────────────────────────────────────┐
        │                   Buyer Advisor Workflow                    │
        └─────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │    Router    │
                              │   (Intent)   │
                              └──────┬───────┘
                                     │
              ┌──────────┬───────────┼───────────┬──────────┐
              ▼          ▼           ▼           ▼          ▼
        ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
        │  Filter  ││Valuation ││Comparison││  Market  ││ Favorites│
        │  Agent   ││  Agent   ││  Agent   ││ Insights ││  Agent   │
        └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
             │           │           │           │           │
             └───────────┴───────────┼───────────┴───────────┘
                                     │
                                     ▼
                          ┌──────────────────┐
                          │   Conversation   │
                          │     Manager      │
                          └────────┬─────────┘
                                   │
                                   ▼
                              ┌─────────┐
                              │   END   │
                              └─────────┘
        """
        return diagram


# Singleton workflow instance
_workflow: Optional[BuyerAdvisorWorkflow] = None


def create_workflow() -> BuyerAdvisorWorkflow:
    """
    Create a new workflow instance.
    
    Returns:
        BuyerAdvisorWorkflow instance
    """
    return BuyerAdvisorWorkflow()


def get_workflow() -> BuyerAdvisorWorkflow:
    """
    Get or create the singleton workflow instance.
    
    Returns:
        BuyerAdvisorWorkflow singleton
    """
    global _workflow
    
    if _workflow is None:
        _workflow = create_workflow()
    
    return _workflow


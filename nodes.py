"""
Node definitions for the LangGraph workflow.
"""
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from state import AgentState
from agents.retrieval_agent import RetrievalAgent
from agents.verification_agent import VerificationAgent
from agents.response_agent import ResponseAgent


def create_retrieval_node(retrieval_agent: RetrievalAgent):
    """Create the retrieval node."""
    def retrieval_node(state: AgentState) -> Dict[str, Any]:
        """Node for retrieving relevant information."""
        print(f"\n--- RETRIEVAL NODE ---")
        print(f"Query: {state['query']}")
        
        result = retrieval_agent.retrieve(state)
        
        print(f"Retrieved {len(result['retrieved_texts'])} texts and {len(result['retrieved_images'])} images")
        print(f"Retrieval successful: {result['retrieval_successful']}")
        
        return result
    
    return retrieval_node


def create_verification_node(verification_agent: VerificationAgent):
    """Create the verification node."""
    def verification_node(state: AgentState) -> Dict[str, Any]:
        """Node for verifying retrieved information."""
        print(f"\n--- VERIFICATION NODE ---")
        
        result = verification_agent.verify(state)
        
        print(f"Verification Status: {result['verification_status']}")
        print(f"Confidence Score: {result['confidence_score']:.2f}")
        
        return result
    
    return verification_node


def create_response_node(response_agent: ResponseAgent):
    """Create the response node."""
    def response_node(state: AgentState) -> Dict[str, Any]:
        """Node for generating final response."""
        print(f"\n--- RESPONSE NODE ---")
        
        result = response_agent.respond(state)
        
        print(f"Generated response with {len(result['sources'])} sources")
        
        return result
    
    return response_node


def input_node(state: AgentState) -> Dict[str, Any]:
    """Initial node to process user input."""
    print(f"\n--- INPUT NODE ---")
    print(f"Processing query: {state['query']}")
    
    return {
        "messages": [HumanMessage(content=f"Query: {state['query']}")],
        "retrieved_texts": [],
        "retrieved_images": [],
        "retrieval_successful": False,
        "verification_status": "pending",
        "verification_notes": "",
        "confidence_score": 0.0,
        "final_answer": "",
        "sources": [],
        "iterations": 0,
        "errors": [],
        "next_step": "retrieve"
    }


def should_continue(state: AgentState) -> str:
    """Determine the next step in the workflow."""
    next_step = state.get("next_step", "end")
    
    if next_step == "end":
        return "end"
    elif next_step == "verify":
        return "verify"
    elif next_step == "respond":
        return "respond"
    elif next_step == "retrieve":
        return "retrieve"
    else:
        return "end"
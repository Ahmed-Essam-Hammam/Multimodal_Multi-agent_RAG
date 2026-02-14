"""
State definitions for the LangGraph workflow.
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State object for the multi-agent RAG system."""

    # User input
    query: str
    query_type: str  # "text" or "image"
    query_image: Optional[bytes]

    # Processing flags
    messages: Annotated[List[BaseMessage], operator.add]

    # Retrieval results
    retrieved_texts: List[Dict[str, Any]]
    retrieved_images: List[Dict[str, Any]]
    retrieval_successful: bool

    # Verification results
    verification_status: str  # "verified", "unverified", "hallucination_detected"
    verification_notes: str
    confidence_score: float

    # Final output
    final_answer: str
    sources: List[Dict[str, Any]]

    # Metadata
    iterations: int
    errors: List[str]

    # Control flow
    next_step: str
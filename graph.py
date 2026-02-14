"""
LangGraph workflow definition for the multimodal RAG system.
"""
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from state import AgentState
from agents import RetrievalAgent, VerificationAgent, ResponseAgent
from tools import RetrievalTools
from utils import VectorStoreManager
from nodes import (
    input_node,
    create_retrieval_node,
    create_verification_node,
    create_response_node,
    should_continue
)
from config import settings


class MultimodalRAGGraph:
    """LangGraph workflow for multimodal RAG with verification."""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url="https://api.cerebras.ai/v1",
            model=settings.MODEL_NAME,
            api_key=settings.CEREBRAS_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        
        # Initialize components
        self.vector_store = VectorStoreManager()
        self.retrieval_tools = RetrievalTools(self.vector_store)
        
        # Initialize agents
        self.retrieval_agent = RetrievalAgent(self.llm, self.retrieval_tools)
        self.verification_agent = VerificationAgent(self.llm)
        self.response_agent = ResponseAgent(self.llm)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("input", input_node)
        workflow.add_node("retrieve", create_retrieval_node(self.retrieval_agent))
        workflow.add_node("verify", create_verification_node(self.verification_agent))
        workflow.add_node("respond", create_response_node(self.response_agent))
        
        # Define edges
        workflow.set_entry_point("input")
        workflow.add_edge("input", "retrieve")
        
        workflow.add_conditional_edges(
            "retrieve",
            should_continue,
            {
                "verify": "verify",
                "respond": "respond",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "verify",
            should_continue,
            {
                "respond": "respond",
                "end": END
            }
        )
        
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def run(self, query: str, query_type: str = "text", query_image: bytes = None) -> dict:
        """Run the RAG workflow."""
        initial_state = {
            "query": query,
            "query_type": query_type,
            "query_image": query_image,
            "messages": [],
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
        
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    def get_vector_store(self) -> VectorStoreManager:
        """Get the vector store manager."""
        return self.vector_store
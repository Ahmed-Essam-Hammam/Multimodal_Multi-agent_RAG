"""
Retrieval agent for finding relevant information from the vector store.
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from tools.retrieval_tools import RetrievalTools
from config import settings

class RetrievalAgent:
    """Agent responsible for retrieving relevant information."""
    
    def __init__(self, llm: ChatOpenAI, retrieval_tools: RetrievalTools):
        self.llm = llm
        self.retrieval_tools = retrieval_tools
        self.system_prompt = """You are a retrieval agent in a multimodal RAG system.
Your job is to retrieve the most relevant text chunks and images based on the user's query.

Instructions:
1. Analyze the user's query to understand what information they're looking for
2. Retrieve relevant text chunks from the knowledge base
3. Retrieve relevant images if the query might benefit from visual information
4. Return a summary of what was found
5. If no relevant information is found, clearly state that

Be precise and focused on finding the most relevant information."""

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Execute the retrieval process."""

        query = state["query"]

        results = self.retrieval_tools.retrieve_multimodal(
            query=query,
            text_top_k=settings.TOP_K_RESULTS,
            image_top_k=3
        )

        retrieved_texts = results["texts"]
        retrieved_images = results["images"]
        retrieval_successful = len(retrieved_texts) > 0 or len(retrieved_images) > 0

        if retrieval_successful:
            summary = self._generate_retrieval_summary(query, retrieved_texts, retrieved_images)
        else:
            summary = "No relevant information found in the knowledge base."

        return {
            "retrieved_texts": retrieved_texts,
            "retrieved_images": retrieved_images,
            "retrieval_successful": retrieval_successful,
            "messages": [HumanMessage(content=f"Retrieval Summary: {summary}")],
            "next_step": "verify" if retrieval_successful else "respond"
        }
    

    def _generate_retrieval_summary(self, query: str, texts: list, images: list) -> str:
        """Generate a summary of what was retrieved."""
        summary_parts = []
        
        if texts:
            summary_parts.append(f"Found {len(texts)} relevant text chunks")
            top_snippets = [t["content"][:100] + "..." for t in texts[:3]]
            summary_parts.append("Top snippets: " + " | ".join(top_snippets))
        
        if images:
            summary_parts.append(f"Found {len(images)} relevant images")
            sources = [img["metadata"].get("source_doc", "unknown") for img in images]
            summary_parts.append(f"Image sources: {', '.join(set(sources))}")
        
        return " | ".join(summary_parts)
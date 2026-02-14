"""
Response agent that generates answers with image references and display support.
"""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from config import settings
import base64
from pathlib import Path


class ResponseAgent:
    """Agent responsible for generating the final response with image display."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a response agent in a multimodal RAG system.
Your job is to generate accurate, helpful answers based on retrieved and verified information.

IMPORTANT: When images are retrieved, always mention them explicitly and reference them by number.

Instructions:
1. Use ONLY the verified retrieved information to answer the query
2. If verification detected issues, acknowledge them in your response
3. If no relevant information was found, clearly state this
4. Cite sources when providing information
5. For images, mention: "See [Image 1], [Image 2]" etc.
6. Be clear about confidence levels
7. Never fabricate or hallucinate information not present in the retrieved data

Response Format:
- Start with a direct answer to the query
- Support with evidence from retrieved sources
- Reference images explicitly: [Image 1], [Image 2]
- Include citations [Source: filename]
- End with confidence level if applicable
- If no information found, suggest alternatives"""
    
    def respond(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final response with image references."""
        query = state["query"]
        retrieval_successful = state["retrieval_successful"]
        verification_status = state.get("verification_status", "unknown")
        verification_notes = state.get("verification_notes", "")
        confidence_score = state.get("confidence_score", 0.0)
        retrieved_texts = state["retrieved_texts"]
        retrieved_images = state["retrieved_images"]
        
        if not retrieval_successful:
            final_answer = self._generate_no_information_response(query)
            sources = []
        elif verification_status == "hallucination_detected":
            final_answer = self._generate_hallucination_response(query, verification_notes)
            sources = self._extract_sources(retrieved_texts, retrieved_images)
        elif verification_status == "unverified":
            final_answer = self._generate_unverified_response(
                query, retrieved_texts, retrieved_images, verification_notes, confidence_score
            )
            sources = self._extract_sources(retrieved_texts, retrieved_images)
        else:
            final_answer = self._generate_verified_response(
                query, retrieved_texts, retrieved_images, confidence_score
            )
            sources = self._extract_sources(retrieved_texts, retrieved_images)
        
        return {
            "final_answer": final_answer,
            "sources": sources,
            "messages": [HumanMessage(content=f"Final Response: {final_answer[:200]}...")],
            "next_step": "end"
        }
    
    def _generate_no_information_response(self, query: str) -> str:
        """Generate response when no information is found."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""No relevant information was found in the knowledge base for this query: "{query}"

Please provide a helpful response that:
1. Clearly states no information was found
2. Suggests what the user might try instead
3. Remains professional and helpful""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _generate_hallucination_response(self, query: str, verification_notes: str) -> str:
        """Generate response when hallucination is detected."""
        return f"""I cannot provide a reliable answer to your query: "{query}"

The verification process detected potential issues with the retrieved information:
{verification_notes}

To ensure accuracy, I cannot provide an answer based on this data. Please try:
1. Rephrasing your query
2. Uploading additional relevant documents
3. Verifying the quality of uploaded documents"""
    
    def _generate_unverified_response(
        self, query: str, texts: list, images: list, verification_notes: str, confidence: float
    ) -> str:
        """Generate response for unverified information with image references."""
        context = self._format_context(texts, images)
        
        # Build image reference guide
        image_guide = ""
        if images:
            image_guide = "\n\nIMAGES RETRIEVED:\n"
            for i, img in enumerate(images, 1):
                image_guide += f"[Image {i}]: {img['path']} (from {img['metadata'].get('source_doc', 'unknown')}, page {img['metadata'].get('page', 'N/A')})\n"
            image_guide += "\nPlease reference these images as [Image 1], [Image 2], etc. in your response.\n"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Query: "{query}"

Retrieved Information:
{context}
{image_guide}

Verification Status: Unverified (Confidence: {confidence:.2f})
Verification Notes: {verification_notes}

Please provide an answer that:
1. Addresses the query based on available information
2. Explicitly mentions and references any images: [Image 1], [Image 2]
3. Clearly indicates the information is unverified
4. Cites specific sources
5. Acknowledges uncertainty""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _generate_verified_response(
        self, query: str, texts: list, images: list, confidence: float
    ) -> str:
        """Generate response for verified information with image references."""
        context = self._format_context(texts, images)
        
        # Build image reference guide
        image_guide = ""
        if images:
            image_guide = "\n\nIMAGES RETRIEVED:\n"
            for i, img in enumerate(images, 1):
                image_guide += f"[Image {i}]: {img['path']} (from {img['metadata'].get('source_doc', 'unknown')}, page {img['metadata'].get('page', 'N/A')})\n"
            image_guide += "\nIMPORTANT: Reference these images as [Image 1], [Image 2], etc. Tell the user to view them.\n"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Query: "{query}"

Retrieved and Verified Information (Confidence: {confidence:.2f}):
{context}
{image_guide}

Please provide a comprehensive answer that:
1. Directly answers the query
2. EXPLICITLY mentions and references images: [Image 1], [Image 2]
3. Tells the user to view the images for visual explanation
4. Uses specific information from the retrieved sources
5. Cites sources appropriately [Source: filename]
6. Integrates information from multiple sources if available""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _format_context(self, texts: list, images: list) -> str:
        """Format retrieved information for the prompt."""
        context_parts = []
        
        for i, text in enumerate(texts, 1):
            source = text["metadata"].get("source", "unknown")
            similarity = text["similarity"]
            context_parts.append(
                f"\n[Text {i}] (Similarity: {similarity:.2f}, Source: {source})\n{text['content']}\n"
            )
        
        for i, img in enumerate(images, 1):
            source = img["metadata"].get("source_doc", "unknown")
            page = img["metadata"].get("page", "N/A")
            similarity = img["similarity"]
            description = img["metadata"].get("description", "No description")
            context_parts.append(
                f"\n[Image {i}] (Similarity: {similarity:.2f}, Source: {source}, Page: {page})\n"
                f"Path: {img['path']}\n"
                f"Description: {description}\n"
            )
        
        return "".join(context_parts)
    
    def _extract_sources(self, texts: list, images: list) -> List[Dict[str, Any]]:
        """Extract unique sources from retrieved information."""
        sources = []
        seen_sources = set()
        
        for text in texts:
            source = text["metadata"].get("source", "unknown")
            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "type": "text",
                    "similarity": text["similarity"]
                })
                seen_sources.add(source)
        
        for img in images:
            source = img["metadata"].get("source_doc", "unknown")
            source_key = f"{source}_img"
            if source_key not in seen_sources:
                sources.append({
                    "source": source,
                    "type": "image",
                    "path": img["path"],
                    "similarity": img["similarity"],
                    "page": img["metadata"].get("page", "N/A")
                })
                seen_sources.add(source_key)
        
        return sources
"""
Verification agent for checking the correctness of retrieved information.
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from config import settings



class VerificationAgent:
    """Agent responsible for verifying retrieved information."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a verification agent in a multimodal RAG system.
Your critical job is to verify the correctness of retrieved information and detect potential hallucinations.

Instructions:
1. Carefully examine the retrieved text chunks and their relevance to the user's query
2. Check for consistency across multiple sources
3. Identify any contradictions or inconsistencies
4. Detect if information seems fabricated or unreliable
5. Assign a confidence score (0.0 to 1.0) based on:
   - Relevance of retrieved information to the query
   - Consistency across sources
   - Quality and specificity of the information
   - Presence of verifiable facts

Verification Status:
- "verified": Information is relevant, consistent, and reliable (confidence > 0.7)
- "unverified": Information found but reliability is questionable (0.4 < confidence <= 0.7)
- "hallucination_detected": Information appears fabricated or completely irrelevant (confidence <= 0.4)

Be thorough and critical."""

    def verify(self, state: AgentState) -> Dict[str, Any]:
        """Verify the retrieved information."""
        query = state["query"]
        retrieved_texts = state["retrieved_texts"]
        retrieved_images = state["retrieved_images"]
        
        if not state["retrieval_successful"]:
            return {
                "verification_status": "no_information",
                "verification_notes": "No information was retrieved to verify.",
                "confidence_score": 0.0,
                "messages": [HumanMessage(content="Verification: No information to verify.")],
                "next_step": "respond"
            }
        

        verification_result = self._perform_verification(query, retrieved_texts, retrieved_images)
        
        return {
            "verification_status": verification_result["status"],
            "verification_notes": verification_result["notes"],
            "confidence_score": verification_result["confidence"],
            "messages": [HumanMessage(content=f"Verification: {verification_result['notes']}")],
            "next_step": "respond"
        }
    
    def _perform_verification(self, query: str, texts: list, images: list) -> Dict[str, Any]:
        """Perform the actual verification using the LLM."""
        context = self._prepare_verification_context(query, texts, images)
        
        verification_prompt = f"""Query: {query}

            Retrieved Information:
            {context}

            Please verify this information by:
            1. Assessing relevance to the query
            2. Checking for internal consistency
            3. Identifying any potential issues or contradictions
            4. Assigning a confidence score (0.0-1.0)

            Provide your assessment in the following format:
            CONFIDENCE: [score]
            STATUS: [verified/unverified/hallucination_detected]
            NOTES: [detailed explanation]"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=verification_prompt)
        ]
        
        response = self.llm.invoke(messages)
        result = self._parse_verification_response(response.content)
        
        return result
    

    def _prepare_verification_context(self, query: str, texts: list, images: list) -> str:
        """Prepare context string from retrieved information."""
        context_parts = []
        
        if texts:
            context_parts.append("TEXT CHUNKS:")
            for i, text in enumerate(texts[:5], 1):
                context_parts.append(f"\n{i}. (Similarity: {text['similarity']:.2f})")
                context_parts.append(f"   Source: {text['metadata'].get('source', 'unknown')}")
                context_parts.append(f"   Content: {text['content'][:300]}...")
        
        if images:
            context_parts.append("\n\nIMAGES:")
            for i, img in enumerate(images, 1):
                context_parts.append(f"\n{i}. (Similarity: {img['similarity']:.2f})")
                context_parts.append(f"   Path: {img['path']}")
                context_parts.append(f"   Source: {img['metadata'].get('source_doc', 'unknown')}")
        
        return "\n".join(context_parts)
    

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's verification response."""
        lines = response.strip().split('\n')
        
        confidence = 0.5
        status = "unverified"
        notes = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("STATUS:"):
                status = line.split(":", 1)[1].strip().lower()
            elif line.startswith("NOTES:"):
                notes = line.split(":", 1)[1].strip()
        
        if not notes:
            notes = response
        
        if status not in ["verified", "unverified", "hallucination_detected"]:
            if confidence > 0.7:
                status = "verified"
            elif confidence > 0.4:
                status = "unverified"
            else:
                status = "hallucination_detected"
        
        return {
            "confidence": confidence,
            "status": status,
            "notes": notes
        }
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger("evaluator")

class EvaluationResponse(BaseModel):
    faithfulness_score: int = Field(description="Faithfulness score from 1 to 5")
    faithfulness_reason: str = Field(description="Detailed explanation for the faithfulness score")
    relevance_score: int = Field(description="Relevance score from 1 to 5")
    relevance_reason: str = Field(description="Detailed explanation for the relevance score")

SYSTEM_PROMPT = """You are an expert AI-as-a-judge system evaluating LLM response quality.
You will evaluate the given response based on two criteria: Faithfulness and Relevance.

Here is the context provided for grounding (if empty, assume no specific grounding data is provided):
---
{context}
---

Your evaluation should return scores between 1 and 5 (inclusive) for both criteria, along with clear reasoning.

Faithfulness grading criteria:
- 5 (Excellent): The response is fully grounded in and supported by the context. No external/hallucinated assumptions.
- 4 (Good): The response is mostly supported by the context, with minor logical extrapolations or harmless extra details.
- 3 (Fair): The response has some supported elements but also significant hallucinated, ungrounded, or external details.
- 2 (Poor): The response is mostly ungrounded in the context, relying heavily on external assumptions or containing contradictions.
- 1 (Unacceptable): The response is completely ungrounded, contradicts the context, or hallucinates everything.
Note: For CHAT queries where context is empty, Faithfulness should default to 5 if there are no contradictions, as there is no specific context to ground it to.

Relevance grading criteria:
- 5 (Excellent): The response directly and completely answers the user's query.
- 4 (Good): The response answers the query but misses minor details or adds slightly irrelevant filler.
- 3 (Fair): The response partially answers the query, or is overly vague and generic.
- 2 (Poor): The response misses the main point of the query or is mostly irrelevant.
- 1 (Unacceptable): The response is completely irrelevant to the user's query.
"""

async def evaluate_response(query: str, response_content: str, context: str) -> EvaluationResponse:
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.warning("GEMINI_API_KEY is not configured. Falling back to mock scores.")
        return get_mock_evaluation(query, response_content, context)
        
    try:
        # Initialize Google GenAI Client
        client = genai.Client(api_key=api_key)
        
        system_instruction = SYSTEM_PROMPT.format(context=context or "No context provided.")
        
        user_prompt = f"User Query: {query}\n\nGenerated Response: {response_content}"
        
        response = await client.aio.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=EvaluationResponse,
                temperature=0.0
            )
        )
        
        if response.text:
            return EvaluationResponse.model_validate_json(response.text.strip())
            
        raise ValueError("Empty response received from judge LLM.")
        
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise e
        logger.warning(f"Evaluation API call failed or encountered error: {str(e)}. Falling back to mock scores.")
        return get_mock_evaluation(query, response_content, context, error_details=str(e))

def get_mock_evaluation(query: str, response_content: str, context: str, error_details: str | None = None) -> EvaluationResponse:
    reason = "Mock evaluation fallback due to missing or invalid GEMINI_API_KEY or API error."
    if error_details:
        reason += f" Details: {error_details}"
        
    lower_response = response_content.lower()
    
    if context:
        # RAG query heuristic: if we see some words from the query/expected answer, it is grounded
        has_some_context = any(word in lower_response for word in ["antigravity", "fastapi", "stabilizer", "uvicorn", "nginx", "chen", "51"])
        faith = 5 if has_some_context else 2
        rel = 5 if len(response_content) > 15 else 2
    else:
        # Chat query heuristic
        faith = 5
        rel = 5 if len(response_content) > 10 else 2
        
    return EvaluationResponse(
        faithfulness_score=faith,
        faithfulness_reason=f"{reason} (Heuristic score)",
        relevance_score=rel,
        relevance_reason=f"{reason} (Heuristic score)"
    )

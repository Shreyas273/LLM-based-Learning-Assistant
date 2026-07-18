from services.llm.llm_router import LLMRouter

_llm = LLMRouter()

def fetch_from_web(question):
    prompt = f"""
    Answer this question using verified educational sources
    like textbooks, official documentation, or reputed websites.
    Keep it factual and concise.

    Question:
    {question}
    """
    resp = _llm.generate(prompt, temperature=0.2, max_tokens=700, top_p=0.9, top_k=40, seed=42)
    return resp.text
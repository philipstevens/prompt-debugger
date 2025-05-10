import openai
import tiktoken
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function: Create reusable OpenAI client
def create_client(api_key):
    return openai.OpenAI(api_key=api_key)

# Function: Run prompt and return output
def run_prompt(
    client,
    prompt,
    model="gpt-3.5-turbo",
    max_tokens=10,
    temperature=0.0,
    top_p=1.0,
    stop=None
):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop
    )
    return response.choices[0].message.content

def format_scaled_cost(cost):
    if cost < 0.001:
        microdollars = cost * 1_000_000
        return f"{microdollars:.0f} µ$"
    elif cost < 0.01:
        millidollars = cost * 1000
        return f"{millidollars:.1f} m$"
    else:
        return f"${cost:.4f}"

# Function: Token and cost calculation
def token_info(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = len(enc.encode(text))

    model_prices = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4-turbo": 0.010,
        "gpt-4": 0.030
    }

    if model not in model_prices:
        raise ValueError(f"Unknown model: {model}. Supported models: {', '.join(model_prices.keys())}")

    cost_per_1k = model_prices[model]
    cost = (tokens / 1000) * cost_per_1k

    return tokens, cost


def get_embedding(text, client):
    return np.array(client.embeddings.create(model="text-embedding-ada-002", input=text).data[0].embedding)

def _similarity_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    sim_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(sim_score * 100, 2)

def _similarity_embeddings(text1, text2, client):
    emb1 = get_embedding(text1, client)
    emb2 = get_embedding(text2, client)
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return round(cosine_sim * 100, 2)

def _similarity_llm(text1, text2, client, model="gpt-3.5-turbo"):
    prompt = f"""
    Rate the semantic similarity between these two texts on a scale of 0 to 100.
    Consider meaning, tone, and style.
    
    Text 1:
    {text1}
    
    Text 2:
    {text2}
    
    Respond with only the number.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return None

def similarity_label(percentage):
    """Return a qualitative label based on similarity percentage."""
    if percentage >= 90:
        return "Nearly Identical"
    elif percentage >= 70:
        return "Very Similar"
    elif percentage >= 50:
        return "Somewhat Similar"
    elif percentage >= 30:
        return "Weak Similarity"
    else:
        return "Very Different"

def compare_similarity(text1, text2, method="tfidf", client=None):
    """
    Compare two texts using the selected similarity method.
    Supported methods: "tfidf", "embeddings", "llm"
    Returns: (percentage, label)
    """
    if method == "tfidf":
        percentage = _similarity_tfidf(text1, text2)
    elif method == "embeddings":
        if client is None:
            raise ValueError("Client is required for embedding similarity.")
        percentage = _similarity_embeddings(text1, text2, client)
    elif method == "llm":
        if client is None:
            raise ValueError("Client is required for LLM similarity.")
        percentage = _similarity_llm(text1, text2, client)
    else:
        raise ValueError(f"Unsupported method: {method}")

    label = similarity_label(percentage)
    return percentage, label



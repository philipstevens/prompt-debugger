import openai
import tiktoken
import difflib

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

# Function: Compare outputs
def compare_outputs(out1, out2):
    # Simulate a diff
    return "\n".join(difflib.unified_diff(out1.splitlines(), out2.splitlines()))

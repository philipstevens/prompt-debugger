import difflib

def run_prompt(prompt, model="gpt-3.5-turbo"):
    # Dummy output that pretends to process the prompt
    return f"[Dummy response to]: {prompt.strip()[:40]}..."

def token_info(prompt, model):
    # Fake token count and cost calculation
    tokens = len(prompt.split()) * 2  # Simulated token estimate
    cost_per_1k = 0.002 if model == "gpt-3.5-turbo" else 0.06
    cost = tokens / 1000 * cost_per_1k
    return tokens, cost

def compare_outputs(out1, out2):
    # Simulate a diff
    return "\n".join(difflib.unified_diff(out1.splitlines(), out2.splitlines()))

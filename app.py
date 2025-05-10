import streamlit as st
from core import create_client, run_prompt, token_info, compare_similarity

st.set_page_config(page_title="Prompt Debugger", page_icon="🛠️", layout="wide")

st.title("Prompt Debugger")

# API Key Management with Clean Startup UX
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "key_confirmed" not in st.session_state:
    st.session_state.key_confirmed = False

# Show Key Input If Not Confirmed Yet (Clean Splash)
if not st.session_state.key_confirmed:
    st.markdown("### 🔑 Enter Your OpenAI API Key to Start")
    st.text_input("API Key", type="password", key="api_key_temp")
    if st.button("Continue"):
        if st.session_state.api_key_temp:
            st.session_state.api_key = st.session_state.api_key_temp
            st.session_state.key_confirmed = True
            st.rerun()
        else:
            st.warning("Please enter your OpenAI API Key.")
    st.stop()

# Pre-filled example prompts
example_prompt1 = "Explain the concept of gravity to a 10-year-old."
example_prompt2 = "Describe gravity using a short analogy."

st.markdown("### Prompt Configuration")
prompt1 = st.text_area("Prompt A", value=example_prompt1, height=150)
prompt2 = st.text_area("Prompt B", value=example_prompt2, height=150)

# Additional Controls
st.markdown("### Advanced Settings")

# Model Selection
model_choice = st.selectbox(
    "Model",
    ["gpt-3.5-turbo (Faster, Cheaper)", "gpt-4-turbo (Higher Quality)", "gpt-4 (Highest Quality, Expensive)"],
    index=0,
    help="Choose the model used for running prompts and LLM similarity"
)
st.write(f"DEBUG: Selected Model = {model_choice}")
model = model_choice.split()[0]  # 'gpt-3.5-turbo', 'gpt-4-turbo', or 'gpt-4'

# Tokens/Temperature
max_tokens = st.slider("Max Tokens (response length)", min_value=1, max_value=1000, value=50)
temperature = st.slider("Temperature (creativity/randomness)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

# Similarity Method
similarity_method = st.selectbox(
    "Similarity Method",
    ["tfidf (Free, Fast)", "embeddings (Low Cost, High Quality)", "llm (High Cost, Highest Quality)"],
    index=0,
    help="Choose how to compare output similarity"
)
method_key = similarity_method.split()[0]  # 'tfidf', 'embeddings', or 'llm'

if st.button("Compare Prompts"):
    client = create_client(st.session_state.api_key)
    with st.spinner("Running prompts..."):
        # Run prompts and capture outputs with user-defined parameters
        out1 = run_prompt(client, prompt1, model, max_tokens=max_tokens, temperature=temperature)
        out2 = run_prompt(client, prompt2, model, max_tokens=max_tokens, temperature=temperature)

        # Token and cost info for input and output separately
        t1_input, c1_input = token_info(prompt1, model)
        t1_output, c1_output = token_info(out1, model)
        t1_total_tokens = t1_input + t1_output
        c1_total_cost = c1_input + c1_output

        t2_input, c2_input = token_info(prompt2, model)
        t2_output, c2_output = token_info(out2, model)
        t2_total_tokens = t2_input + t2_output
        c2_total_cost = c2_input + c2_output

        # Show Outputs
        st.subheader("Output")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Prompt A Output:**")
            st.text_area("Prompt A Output", out1, height=300, key="output_a", disabled=True)
        
        with col2:
            st.markdown("**Prompt B Output:**")
            st.text_area("Prompt B Output", out2, height=300, key="output_b", disabled=True)
    
        # Show Similarity
        similarity_percentage, similarity_label = compare_similarity(out1, out2, method=method_key, client=client)
        st.subheader("Similarity")
        st.markdown(f"{similarity_percentage}% ({similarity_label})")
        
        # Show Token and Cost Breakdown
        st.subheader("Token & Cost Info")
        st.markdown(f"""
        **Prompt A**
        - Input: {t1_input} tokens (${c1_input:.4f})
        - Output: {t1_output} tokens (${c1_output:.4f})
        - Total: {t1_total_tokens} tokens (${c1_total_cost:.4f})

        **Prompt B**
        - Input: {t2_input} tokens (${c2_input:.4f})
        - Output: {t2_output} tokens (${c2_output:.4f})
        - Total: {t2_total_tokens} tokens (${c2_total_cost:.4f})
        """)

# Spacer and Divider before API Key Management
st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### 🔒 API Key Management")
with st.expander("⚙️ Manage API Key (click to expand)", expanded=False):
    new_key = st.text_input("Update API Key", type="password")
    if st.button("Update API Key"):
        if new_key:
            st.session_state.api_key = new_key
            st.success("API Key updated successfully.")
            st.rerun()


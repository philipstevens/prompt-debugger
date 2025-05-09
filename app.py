import streamlit as st
from core import create_client, run_prompt, token_info, compare_outputs

st.set_page_config(page_title="Prompt Debugger", page_icon="🛠️")

st.title("Prompt Debugger")

# API Key Input with Session State in Sidebar
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar.expander("🔑 API Key (required to run)", expanded=True):
    st.text_input("Enter your OpenAI API Key", type="password", key="api_key")

# Live Key Status Indicator in Sidebar
if st.session_state.api_key:
    st.sidebar.success("✅ API Key loaded. Ready to run prompts.")
else:
    st.sidebar.warning("🔑 API Key required to run prompts.")

# Block App If API Key Is Missing
if not st.session_state.api_key:
    st.title("🔑 Enter Your OpenAI API Key to Start")
    st.text_input("API Key", type="password", key="api_key")
    st.stop()

model = "gpt-3.5-turbo"

# Pre-filled example prompts
example_prompt1 = "Explain the concept of gravity to a 10-year-old."
example_prompt2 = "Describe gravity using a short analogy."

st.markdown("### Prompt Configuration")
prompt1 = st.text_area("Prompt A", value=example_prompt1, height=150)
prompt2 = st.text_area("Prompt B", value=example_prompt2, height=150)

# Additional user controls
st.markdown("### Advanced Settings")
max_tokens = st.slider("Max Tokens (response length)", min_value=1, max_value=1000, value=50)
temperature = st.slider("Temperature (creativity/randomness)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

if st.button("Compare Prompts"):
    if not st.session_state.api_key:
        st.error("Please enter your OpenAI API Key.")
    else:
        client = create_client(st.session_state.api_key)
        with st.spinner("Running prompts..."):
            # Run prompts and capture outputs with user-defined parameters
            out1 = run_prompt(client, prompt1, model, max_tokens=max_tokens, temperature=temperature)
            out2 = run_prompt(client, prompt2, model, max_tokens=max_tokens, temperature=temperature)

            # Compare outputs
            diff = compare_outputs(out1, out2)

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
        st.write("**Prompt A Output:**")
        st.code(out1)
        st.write("**Prompt B Output:**")
        st.code(out2)

        # Show Diff
        st.subheader("Diff")
        st.code(diff if diff else "No differences found.")

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

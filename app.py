import streamlit as st
from core import run_prompt, token_info, compare_outputs

st.title("Prompt Debugger")

model = "gpt-3.5-turbo"
prompt1 = st.text_area("Prompt A", height=150)
prompt2 = st.text_area("Prompt B", height=150)

if st.button("Compare Prompts"):
    with st.spinner("Running prompts..."):
        out1 = run_prompt(prompt1, model)
        out2 = run_prompt(prompt2, model)
        diff = compare_outputs(out1, out2)
        t1, c1 = token_info(prompt1, model)
        t2, c2 = token_info(prompt2, model)

    st.subheader("Output")
    st.write("**Prompt A Output:**")
    st.code(out1)
    st.write("**Prompt B Output:**")
    st.code(out2)

    st.subheader("Diff")
    st.code(diff if diff else "No differences found.")

    st.subheader("Token & Cost Info")
    st.markdown(f"""
    - Prompt A: {t1} tokens (${c1:.4f})  
    - Prompt B: {t2} tokens (${c2:.4f})
    """)

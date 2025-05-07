import streamlit as st

st.title("Prompt Debugger (Test Mode)")

st.write("This is a test version of the app. No OpenAI API calls are made.")

prompt1 = st.text_area("Prompt A", height=150)
prompt2 = st.text_area("Prompt B", height=150)

if st.button("Compare Prompts"):
    st.subheader("Output")
    st.write("**Prompt A Output:** This is a fake response for Prompt A.")
    st.write("**Prompt B Output:** This is a fake response for Prompt B.")
    st.subheader("Diff")
    st.code("Sample diff:\n- Old line\n+ New line")
    st.markdown("**Token Count (est.):** 23  \n**Estimated Cost:** $0.00")

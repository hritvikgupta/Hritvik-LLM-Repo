import streamlit as st
from LangChainHelper import create_vector_db, get_qa_chain

st.title("Hritvik QA")
btn = st.button("Hritvik Knowledgebase")
if btn:
    pass

question = st.text_input("Question:")
if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer:")

    # Check if the response contains "I don't know"
    if "i don't know" in response['result'].strip().lower():
        st.write("The answer is unclear. Try rephrasing your question with specific keywords.")
    else:
        st.write(response['result'])

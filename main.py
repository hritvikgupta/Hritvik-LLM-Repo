import streamlit as st
from LangChainHelper import create_vector_db, get_qa_chain

st.title("Hritvik QA")
btn = st.button("Hritvik Knowledgebase")

# Custom CSS for button-like links
button_style = """
<style>
    .button-link {
        display: block; /* Change to block to fill column */
        padding: 0.5em 1em;
        background-color: #000000;
        color: white !important;
        border: 2px solid #FFFFFF;
        border-radius: 4px;
        text-align: center;
        text-decoration: none;
        font-size: 1em;
        transition: background-color 0.3s, color 0.3s;
        margin: 0 auto; /* Center button-link in column */
    }
    .button-link:hover {
        background-color: #00796b;
        color: white;
    }
    .stButton > button {
        width: 100%;
    }
    .stTextInput > div > div > input {
        width: 100%; /* Make text input full width */
    }
    .stMarkdown {
        padding: 0; /* Remove default padding */
    }
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Links Section with Button Style
st.markdown("## Connect with Hritvik Gupta")

resume_url = "https://raw.githubusercontent.com/hritvikgupta/Hritvik-LLM-Repo/main/hritvikresumefeb2024.pdf"
github_url = "https://github.com/hritvikgupta"
linkedin_url = "https://www.linkedin.com/in/hritvik-gupta-link/"
google_scholar_url = "https://scholar.google.com/citations?user=ShxBp2MAAAAJ&hl=en"

# Create a single row for links using equal-width columns
cols = st.columns(4)
links = [resume_url, github_url, linkedin_url, google_scholar_url]
labels = ["Download Resume", "GitHub", "LinkedIn", "Google Scholar"]

for col, url, label in zip(cols, links, labels):
    with col:
        st.markdown(f"<a href='{url}' target='_blank' class='button-link'>{label}</a>", unsafe_allow_html=True)

if btn:
    # Your code for when the button is clicked
    pass

# Input and response section
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

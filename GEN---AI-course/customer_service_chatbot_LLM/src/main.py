import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.set_page_config(page_title="AI Customer Service Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Customer Service AI Assistant")
st.markdown("Welcome! I am an advanced AI trained on our company's FAQs. How can I help you today?")

# Sidebar for admin tasks
with st.sidebar:
    st.header("Admin Settings")
    st.markdown("Use this to generate the Knowledge Base from the CSV data.")
    btn = st.button("🧠 Build Knowledgebase", type="primary")
    if btn:
        with st.spinner("Building vector database..."):
            create_vector_db()
        st.success("Knowledgebase created successfully!")
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Built by Rahul** 🚀")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about our courses or services..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from QA chain
    with st.spinner("Thinking..."):
        try:
            chain = get_qa_chain()
            response = chain(prompt)
            answer = response["result"]
            sources = response.get("source_documents", [])
        except Exception as e:
            answer = f"Error: Please ensure the knowledgebase is built first. ({str(e)})"
            sources = []

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("View Source Context"):
                for doc in sources:
                    st.info(doc.page_content)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

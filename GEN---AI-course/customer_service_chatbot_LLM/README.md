# Generative AI Q&A: Advanced Customer Service Chatbot
**Built by: Rahul** 🚀

This is an end-to-end, state-of-the-art LLM project utilizing Google Palm and LangChain. I built this Q&A system for an e-learning company called Nullclass (which provides data-related courses and virtual internships). By transforming their raw FAQ data into an interactive AI, this system dramatically reduces the workload of human staff by answering student questions instantly and accurately.

## 🌟 Next-Level Features

- **Modern Chat UI**: A sleek, conversational Streamlit interface with persistent chat history.
- **Intelligent RAG System**: Leverages FAISS Vector Database and HuggingFace embeddings (`hkunlp/instructor-large`) to retrieve highly relevant context.
- **Google PaLM Integration**: Uses Google's PaLM LLM for precise, context-aware answers.
- **Source Inspection**: Users can expand the "View Source Context" tab to see exactly which FAQ document the AI used to formulate its answer.
- **Admin Sidebar**: 1-click knowledgebase building directly from the UI.



## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/Rxhulnxyak/GEN-AI-Course.git
```
2.Navigate to the project directory:

```bash
  cd GEN-AI-Course/GEN---AI-course/customer_service_chatbot_LLM
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```
## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter

## Sample Questions
  - Do you guys provide internship and also do you offer EMI payments?
  - Do you have javascript course?
  - Should I learn power bi or tableau?
  - I've a MAC computer. Can I use powerbi on it?
  - I don't see power pivot. how can I enable it?

## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your Google API key.
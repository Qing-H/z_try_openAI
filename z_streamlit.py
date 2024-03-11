import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI


@st.cache_resource
def load_data(vector_store_dir: str = "data/IRM_HELP"):
    db = FAISS.load_local(vector_store_dir, AzureOpenAIEmbeddings())
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    print("Loading data...")

    IRM_ChAT_BOT = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.7}))
    IRM_ChAT_BOT.return_source_documents = True

    return IRM_ChAT_BOT


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    IRM_ChAT_BOT = load_data()

    ans = IRM_ChAT_BOT.invoke({"query": message})
    if ans["source_documents"] or enable_chat:
        return ans["result"]
    else:
        return "Sorry, I don't know ~"

'''
def streamlit_ui():

    st.title('IRM HELP PORTAL')

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner("Generating......"):
            output = chat(prompt, st.session_state["chat_history"])

            st.session_state["chat_answers_history"].append(output)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt, output))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)

'''
if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/z_try_openAI/.env"
    load_dotenv(dotenv_path=env_path, verbose=True)


    st.title('IRM HELP PORTAL')

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner("Generating......"):
            output = chat(prompt, st.session_state["chat_history"])

            st.session_state["chat_answers_history"].append(output)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt, output))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)
import gradio as gr
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI


def initialize_data(vector_store_dir: str = "data/IRM_HELP"):
    db = FAISS.load_local(vector_store_dir, AzureOpenAIEmbeddings())
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    global IRM_ChAT_BOT
    IRM_ChAT_BOT = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.7}))
    IRM_ChAT_BOT.return_source_documents = True

    return IRM_ChAT_BOT


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = IRM_ChAT_BOT({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "Sorry, I don't know."


def launch_ui():
    demo = gr.ChatInterface(
        fn=chat,
        title="IRM HELP PORTAL",
        theme="soft",
        textbox=gr.Textbox(placeholder="Input your question here", container=False, scale=7),
        chatbot=gr.Chatbot(height=600),

    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/z_try_openAI/.env"
    load_dotenv(dotenv_path=env_path, verbose=True)

    initialize_data()
    launch_ui()

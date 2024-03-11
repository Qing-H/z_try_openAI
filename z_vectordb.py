from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA



env_path = os.getenv("HOME") + "/Documents/z_try_openAI/.env"
load_dotenv(dotenv_path=env_path, verbose=True)
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pvg-azure-openai-uk-south.openai.azure.com"

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  api_version="2023-05-15"
)

doc_reader = PdfReader('IRM Help.pdf')
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
print(len(raw_text))

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

print(len(texts))

embeddings = AzureOpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
data_path = "data/IRM_HELP"
docsearch.save_local(data_path)
new_docsearch = FAISS.load_local(data_path, AzureOpenAIEmbeddings())

llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(llm,
             retriever=new_docsearch.as_retriever(search_type="similarity_score_threshold",
               search_kwargs={"score_threshold": 0.7}))
qa_chain.combine_documents_chain.verbose = True
qa_chain.return_source_documents = True


Q1 = "how to Configuring Regional Settings"
print(qa_chain({"query": Q1}))








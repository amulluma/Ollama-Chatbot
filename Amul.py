from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader


# ..............Load the Document.................................
loader = WebBaseLoader("https://sites.google.com/view/amulkhandekar?usp=sharing")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# ...............Embeddings........................................
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
index = FAISS.from_documents(docs,embeddings)


# ...............Saving and Loading................................
save_directory = "Storage amul"
index.save_local(save_directory)
new_db = FAISS.load_local(save_directory,embeddings,allow_dangerous_deserialization=True)


# ...............Defining Retriever................................
retriever = new_db.as_retriever(search_type="similarity",search_kwargs={"k":3})



# ...............Prompt............................................
template = """ Start answering the question like a chatbot and the question based on the provided context. if you do not found relavent result just say this information is not available in the helpcenter. 

Context: {context}
Question : {question}

Helpful Answer :
"""
prompt = PromptTemplate(template=template,input_variables=["context","question"])



# ..............Define LLM Model....................................
llm = Ollama(model="llama3.1:8b",temperature=0.6)

# ..............Define qa_Chain........................................
llm_chain = LLMChain(
                  llm=llm, 
                  prompt=prompt, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )


qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )


# print(qa("What is described inside?")["result"])

def respond(question,history):
    return qa(question)["result"]


gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=200),
    textbox=gr.Textbox(placeholder="Ask me question related to Amul", container=False, scale=7),
    title="Amul Portfolio",
   
    retry_btn=None,

).launch(share = True)

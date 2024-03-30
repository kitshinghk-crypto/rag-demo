#!/usr/bin/env python
# coding: utf-8
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os


# Load PDF
loaders = PyPDFLoader("NIST.SP.800-53r5.pdf")
docs=loaders.load()


# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs[43:399])# read only the core part


# create Chroma db:
embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)

#openai prompt
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"


from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Build prompt
template = """you are a bot that help answer user question based on the document we have provided to you.
You should answer user's question only based on the context provided. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Give detail as much as possible based on your understanding of the context provided.
Always say "thanks for asking!" at the end of the answer. 

Context: {context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


# Run chain
retrieval_qa = RetrievalQA.from_chain_type(llm,
                           retriever=vectordb.as_retriever(),
                           return_source_documents=True,
                           chain_type_kwargs={
                               "prompt": QA_CHAIN_PROMPT,
                               "verbose": False
                           })

# QA:
question="tell me about the guideline regarding the topic of the defination of privacy data. provide the page numbers of the document with respects to your findings."
result = retrieval_qa({"query": question})
print("Q: %s\n"%question)
print("A: %s\n"%result["result"])

question="tell me about the guideline regarding the topic of the defination of multi factor authentication. provide the page numbers of the document with respects to your findings."
result = retrieval_qa({"query": question})
print("Q: %s\n"%question)
print("A: %s\n"%result["result"])

question="tell me about the guideline regarding the topic of the defination of incident response. provide the page numbers of the document with respects to your findings."
result = retrieval_qa({"query": question})
print("Q: %s\n"%question)
print("A: %s\n"%result["result"])

question="tell me about the guideline regarding the topic of the defination of least privilege principle. provide the page numbers of the document with respects to your findings."
result = retrieval_qa({"query": question})
print("Q: %s\n"%question)
print("A: %s\n"%result["result"])


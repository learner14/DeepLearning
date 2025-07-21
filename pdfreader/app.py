from fastapi import FastAPI
import os
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain.text_splitter import CharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader  

os.environ["OPENAI_API_KEY"] = ' Your Key'
os.environ["ACTIVELOOP_TOKEN"] = 'Your key'


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

@app.get("/chat/")
async def root(query:str):
  
    loader = PyPDFLoader("The One Page Linux Manual.pdf")
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(pages)

    embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')

    ## create vector database
 
    db = DeeplakeVectorStore(dataset_path="./my_deeplake/", embedding_function=embeddings, overwrite=True)
    ids = db.add_documents(docs)

    model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)


    # create Prompt template

    template = """You are an exceptional customer support chatbot that gently answer questions.

                you know the following context information.
                {chunks_formatted}
                Answer the following question from a customer from context, if not found then answer from LLM.

                Question:{query}

                Answer:"""
    

    prompt = PromptTemplate(
          input_variables=["chunks_formatted","query"],
          template=template
      )
 

    ## top relevent documents to a specif query
    #query="where is google new plan"
    docs=db.similarity_search(query)
    retrieved_chunks=[doc.page_content for doc in docs]

    # format the prompt
    chunks_formatted="\n\n".join(retrieved_chunks)
    prompt_formatted=prompt.format(chunks_formatted=chunks_formatted,query=query)

    #generate Answer

    model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
    #answer=model(prompt_formatted)
    messages=model.invoke(prompt_formatted)

    return {messages.content}

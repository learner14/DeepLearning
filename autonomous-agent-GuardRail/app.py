from fastapi import FastAPI
import os
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import requests
from  newspaper import Article
import time
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

os.environ["OPENAI_API_KEY"] = 'Your key Here'
os.environ["ACTIVELOOP_TOKEN"] = 'Your Key here'

retriever=""

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

    # This is the function that defines our custom tool that retrieves relevant
    # docs from Deep Lake
def retrieve_n_docs_tool(query: str,) -> str:
    """Searches for relevant documents that may contain the answer to the query."""
    embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')
    db = DeeplakeVectorStore(dataset_path="./my_deeplake/", embedding_function=embeddings, overwrite=True)
    # Get the retriever object from the deep lake db object and set the number
    # of retrieved documents to 3
    retriever = db.as_retriever()
    retriever.search_kwargs['k'] = 3
    # We define some variables that will be used inside our custom tool
    CUSTOM_TOOL_DOCS_SEPARATOR ="\n---------------\n" # how to join together the retrieved docs to form a single string
    docs = retriever.get_relevant_documents(query)
    texts = [doc.page_content for doc in docs]
    texts_merged = "---------------\n" + CUSTOM_TOOL_DOCS_SEPARATOR.join(texts) + "\n---------------"
    return texts_merged


@app.get("/chat/")
def root(query:str):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }   

    article_urls = [
        "https://www.lincolnfinancial.com/public/individuals/products/lifeinsurance/termlife/",
        "https://www.lincolnfinancial.com/public/individuals/products/lifeinsurance/permanentlife/",
        "https://www.lincolnfinancial.com/public/individuals/products/lifeinsurance/permanentlife/indexeduniversallife/",
        "https://www.lincolnfinancial.com/public/individuals/products/longtermcareplanning",
        "https://www.lincolnfinancial.com/public/individuals/products/annuities/fixedannuities",
        "https://www.lincolnfinancial.com/public/individuals/products/annuities/fixedindexedannuities",
        "https://www.lincolnfinancial.com/public/individuals/products/annuities/variableannuities",
        "https://www.lincolnfinancial.com/public/individuals/products/annuities/indexlinkedannuities",
        "https://www.artificialintelligence-news.com/2023/05/15/jay-migliaccio-ibm-watson-on-leveraging-ai-to-improve-productivity/"
       # "https://www.artificialintelligence-news.com/2023/05/15/iurii-milovanov-softserve-how-ai-ml-is-helping-boost-innovation-and-personalisation/",
        #"https://www.artificialintelligence-news.com/2023/04/21/google-creates-new-ai-division-to-challenge-openai/"
    ]
    



    
    session=requests.Session()
    pages_content = [] # where we save the scraped articles
    for url in article_urls:
        try:
            time.sleep(2) # sleep two seconds for gentle scraping
            response = session.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                article = Article(url)
                article.download() # download HTML of webpage
                article.parse() # parse HTML to extract the article text
                pages_content.append({ "url": url, "text": article.text })
            else:
                print(f"Failed to fetch article at {url}")
        except Exception as e:
            print(f"Error occurred while fetching article at {url}: {e}")

    #If an error occurs while fetching an article, we catch the exception and print
    #an error message. This ensures that even if one article fails to download,
    #the rest of the articles can still be processed.


    embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')

    db = DeeplakeVectorStore(dataset_path="./my_deeplake/", embedding_function=embeddings, overwrite=True)
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts = []
    for d in pages_content:
            chunks = text_splitter.split_text(d["text"])
            for chunk in chunks:
                all_texts.append(chunk)

    ids = db.add_texts(all_texts)

    # We create the tool that uses the "retrieve_n_docs_tool" function
    tools = [
        Tool(
            name="Search Private Docs",
            func=retrieve_n_docs_tool,
            description="useful for when you need to answer questions about current events about Artificial Intelligence"
        )
    ]
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    config = RailsConfig.from_path("config")
    guardrails = RunnableRails(config)

    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

    chain_with_guardrails = guardrails | agent
    messages=  chain_with_guardrails.invoke({"input":query})
    if type(messages) is dict:
        response=messages['output']
    else:
        response=messages


    return {response}

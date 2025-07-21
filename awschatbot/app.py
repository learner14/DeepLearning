from fastapi import FastAPI

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrockConverse
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

@app.get("/chat/")
async def root(query:str):
    
    llm = ChatBedrockConverse(
        model="amazon.nova-pro-v1:0",
        temperature=0,
        max_tokens=1000,
        # other params...
        )
    
    
    prompt = ChatPromptTemplate.from_messages(  
            [
                (
                    "system",
                    "You are a helpful assistant. Answer all questions to the best of your ability.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    chain = prompt | llm    
    with_message_history = RunnableWithMessageHistory(chain, get_session_history) 
    config = {"configurable": {"session_id": "abc5"}}
    
    response = with_message_history.invoke(
        [HumanMessage(content=query)],config=config,)
    messages=response.content

    return {messages}




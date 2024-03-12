import boto3
import json
import os
import psycopg2

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import flask
from flask import Flask, request, jsonify
import aws
import db

app = Flask(__name__)
load_dotenv()

def getConnectionString():
    return PGVector.connection_string_from_db_params(
        driver = os.getenv('DRIVER'),
        host = os.getenv('HOST'),
        port = os.getenv('PORT'),
        database = os.getenv('DATABASE'),
        user = os.getenv('USER'),
        password = os.getenv('PASSWORD')
    )

@app.route('/', methods=['GET'])
def getDefault():
    return 'Hello'

@app.route('/querySwinz', methods=['POST'])
def querySwinz():
    
    request_json = json.loads(request.data)
    query = request_json['message']
    session_id = request_json['session_id']
    lang = request_json['lang']
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name="eu-central-1"
    )

    # - create the titan Model
    llm, bedrock_embeddings = aws.load_llm()
    
    #vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    COLLECTION_NAME = "swinz_insurance"
    CONNECTION_STRING = getConnectionString()

    vectorstore = PGVector(
        collection_name = COLLECTION_NAME,
        connection_string = CONNECTION_STRING,
        embedding_function = bedrock_embeddings
    )
    history_list = db.fetch_conversation_log(session_id)
    history = ""
    for row in history_list:
        history = history + row[0] + ":" + row[1] + "\n"

    db.insert_conversation_log(session_id, "Bot", "Human", query)
    prompt_template = """
        You are a helpful AI assistant of Swinz Insurance expert in identifying the relevant topic
        from user's question about Insurance and providing a response in detail. 

        Use the following pieces of context and history to provide an answer to the question at the end. 
        
        {{context}} 
    
        {history}
        Question: {{question}}

    Assistant:"""

    
    prompt_template = prompt_template.format(history = history)
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
    print ('BEFORE RETERIVAL')
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents = True,
        verbose = True,
        #memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True),
        #condense_question_prompt=prompt
        chain_type_kwargs = {
            "prompt": prompt,
            "verbose": True
        #    "memory": ConversationBufferMemory(memory_key="history", input_key="question", chat_memory=history)
        }
    )
    print ('AFTER RETERIVAL')
    result = qa.invoke({"query":query})
    print ('Y RETERIVAL')
    #print(result['source_documents'][0])
    reply_id = db.insert_conversation_log(session_id, "Human", "Bot", result['result'])
    #print(qa)

    response_data = {
        "reply_id": reply_id,
        "answer": result["result"],
        #"source": result["source_documents"][0]
    }

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type':'application/json',
        'Access-Control-Allow-Headers': 'Authorization,Content-Type,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,Keep-Alive,X-Requested-With,If-Modified-Since','Access-Control-Max-Age': '3600','Content-Type':'application/json'
    }
    return jsonify(reply_id=reply_id, answer=result["result"])
    #return response_data

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=5000)

#ai_message = querySwinz("what is the policy on conflicts of interest", '1234')
#history = "Human: what is the policy on conflicts of interest\n Bot:" + ai_message['answer']+ "\n Human: can you repeat that?\nBot:"

#ai_message = querySwinz("can you repeat that?", '1234')
#history = history + ai_message['answer']
#print (history) 

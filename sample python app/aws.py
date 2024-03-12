import boto3
import os
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

def load_llm():
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name="eu-central-1"
    )

    # - create the titan Model
    llm = Bedrock(model_id=os.getenv('CHAT_MODEL'), client=bedrock)
    embeddings = BedrockEmbeddings(model_id=os.getenv('EMBEDDING_MODEL'), client=bedrock)
    return llm, embeddings
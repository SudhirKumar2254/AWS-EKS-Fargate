import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.s3_directory import S3DirectoryLoader
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv

load_dotenv()

def createEmbeddings():
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name="eu-central-1"
    )   
    bedrock_embeddings = BedrockEmbeddings(model_id = os.getenv('EMBEDDING_MODEL'), client=bedrock)

    loader = S3DirectoryLoader(bucket = os.getenv('BUCKET'), prefix='')
    documents = loader.load()
    print("docs loaded from s3")
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 100,
    )
    docs = text_splitter.split_documents(documents)
    print("docs split")
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(docs)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
    print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver = os.getenv('DRIVER'),
        host = os.getenv('HOST'),
        port = os.getenv('PORT'),
        database = os.getenv('DATABASE'),
        user = os.getenv('USER'),
        password = os.getenv('PASSWORD')
    )
   
    COLLECTION_NAME = "swinz_insurance"
    print("db connection made")
    db = PGVector.from_documents(
        embedding = bedrock_embeddings,
        documents = docs,
        collection_name = COLLECTION_NAME,
        connection_string = CONNECTION_STRING
    )
    print("vector created")
    
createEmbeddings()
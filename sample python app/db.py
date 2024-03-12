
import os
import psycopg2
import psycopg2.pool 
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv

load_dotenv()
pool = psycopg2.pool.SimpleConnectionPool(1, 3, host=os.getenv('HOST'), port=os.getenv('PORT'), database=os.getenv('DATABASE'), user=os.getenv('USER'), password=os.getenv('PASSWORD'))
connection = pool.getconn()

def insert_conversation_log(session_id, to_id, from_id, message):
    cur = connection.cursor()
    sql = ('insert into conversation_details (session_id, to_id, from_id, message) VALUES (%s,%s,%s,%s) RETURNING id')
    records = (session_id, to_id, from_id, message)
    cur.execute(sql, records)
    reply_id = cur.fetchone()[0]
    connection.commit()
    return reply_id

def fetch_conversation_log(session_id):
    cur = connection.cursor()
    cur.execute("Select from_id, message from conversation_details where session_id=%s order by created", [session_id])
    result = cur.fetchall()
    print(result)
    return result

fetch_conversation_log('1234')
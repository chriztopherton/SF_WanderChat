import streamlit as st 
import time 
import random 
from openai import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
import re
import requests
from typing import Literal
import os 
from dotenv import load_dotenv

# from utils.self_rag import * 
from utils.UI import *

load_dotenv()

st.set_page_config(page_title="WanderChat", page_icon=":speech_balloon:",layout="wide")
# add_logo("./static/San_Jose_State_Spartans_logo.png","WanderChat, a context-aware travel chatbot.")

# st.sidebar.title("Custom model")

if 'user_hf_token' not in st.session_state: st.session_state['user_hf_token'] = ''
if 'model_base_url' not in st.session_state: st.session_state['model_base_url'] = ''

hf_token = os.getenv("hf_token")

# pinecone = Pinecone()



# db = FAISS.load_local("./funcheap_2024-04-26_2024-06-25_db",
#                       HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}),
#                       allow_dangerous_deserialization=True).as_retriever()
# st.session_state['retriever'] = db


# with st.sidebar:
    # with st.expander("HF token"):
    #     # user_hf_token = st.text_input("","",key="hf_key")
    #     st.session_state['user_hf_token'] = st.secrets['hf_token']
        
    # with st.expander("LLM Inference Endpoint"):
    #     # model_base_url = st.text_input("","",key="model_key")
    #     st.session_state['model_base_url'] = st.secrets['model_base_url']
        

system_prompt = '''Answer the question as if you are a travel agent and your goal is to provide excellent customer service and to provide
        personalized travel recommendations with reasonings based on their question. Do not repeat yourself or include any links or HTML.'''
        
index_name = 'wanderchat-travel-advisory-rag'

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Pinecone.from_existing_index(
    index_name, embed_model.embed_query)

headers = {"Accept" : "application/json","Authorization": f"Bearer {st.secrets['hf_token']}","Content-Type": "application/json" }
def query(payload):
  response = requests.post(st.secrets['model_base_url'], headers=headers, json=payload)
  return response.json()

def find_match(input):
    # input_em = embeddings.encode(input).tolist()
    result = vectorstore.similarity_search(input)
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    return ' '.join([d.page_content for d in result])
    
if st.secrets['hf_token'] and st.secrets['model_base_url']:
    
    chat = ChatOpenAI(model_name="tgi",openai_api_key=st.session_state['user_hf_token'],openai_api_base=st.session_state['model_base_url'],)
    client = OpenAI(base_url=st.session_state['model_base_url'], api_key=st.session_state['user_hf_token'])
    st.session_state['llm'] = chat
    memoryforchat=ConversationBufferMemory()
    convo=ConversationChain(memory=memoryforchat,llm=chat,verbose=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memoryforchat.save_context({"input":message["human"]},{"outputs":message["AI"]})

    if "message" not in st.session_state:
        st.session_state.message = [{"role":"assistant","content":"how may i help you "}]
    for message1 in st.session_state.message:
        with st.chat_message(message1["role"]):
            st.markdown(message1["content"])
            
    if prompt:=st.chat_input("Say Something"):
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.message.append({"role":"user","content":prompt})
                
                #st.sidebar.write(retrieval_grader(prompt))
                
            with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        time.sleep(1)
                        # responce=convo.predict(input=prompt)
                        
                        context = find_match(prompt)
                        
                        prompt = f"""{system_prompt}
                            Context: {context} \n\n
                            Question: {prompt}"""
                        
                        #st.sidebar.write(prompt)

                            
                        input_len = len(prompt.split())
                        max_token_len = 1500-input_len-100 #100 buffer

                        start_time = time.time()
                        while True: #while loop for token
                            answer = query({'inputs': f"<s>[INST] {prompt} [/INST]",
                                        'parameters': {"max_new_tokens": max_token_len}})
                            if 'error' not in answer:
                                break  #exit the while loop if there is no error
                            max_token_len -= 100 #reduce by 100 in while loop
                            print(f"Failed to process prompt with token length: {max_token_len}")
                            if max_token_len <= 0:
                                break
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        print(answer)
                        
                        answer = answer[0]['generated_text'].replace(f"<s>[INST] {prompt} [/INST]","")
                        answer = answer.replace(" . ",". ").strip()
                        answer = re.sub(r'<ANSWER>.*$', '', answer, flags=re.DOTALL) #RAFT specific
                        responce = re.sub(r'Final answer: .*$', '', answer, flags=re.DOTALL) #RAFT specific
                        

                                                # chat_completion = client.chat.completions.create(
                        #     model="tgi",messages=[{"role": "user","content": f"Context:\n {context} \n\n Query:\n{prompt}"}],stream=False,max_tokens=1350)
                        #     # model="tgi",messages=[{"role": "user","content": prompt}],stream=False,max_tokens=1300)
                        # responce = chat_completion.choices[0].message.content
                        
                        # responce = responce.replace(" . ",". ").strip()
                        # responce = re.sub(r'<ANSWER>.*$', '', responce, flags=re.DOTALL) #RAFT specific
                        # responce = re.sub(r'Final answer: .*$', '', responce, flags=re.DOTALL) #RAFT specific
                        
                        st.write(responce)
                        
            st.session_state.message.append({"role":"assistant","content":responce})
            message={'human':prompt,"AI":responce}
            st.session_state.chat_history.append(message)
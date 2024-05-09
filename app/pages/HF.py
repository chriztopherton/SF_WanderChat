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
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.retrievers import EnsembleRetriever
import re
import requests
from typing import Literal
import os 
from dotenv import load_dotenv
from utils.render_image import *

from utils.UI import *

load_dotenv()

st.set_page_config(page_title="WanderChat", page_icon=":speech_balloon:",layout="wide")

if 'user_hf_token' not in st.session_state: st.session_state['user_hf_token'] = ''
if 'model_base_url' not in st.session_state: st.session_state['model_base_url'] = ''
if "message" not in st.session_state: st.session_state.message = [{"role":"assistant","content":"Hello, I am Wanderchat. How may I assist you?"}]
if 'user_feedback' not in st.session_state: st.session_state['user_feedback'] = []
if 'model_responded' not in st.session_state: st.session_state['model_responded'] = False

hf_token = os.getenv("hf_token")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


index_dict = {'Reddit':'wanderchat-reddit-rag',
              'SF Bay Area Activities':'wanderchat-funcheap-rag',
              'U.S. Travel Advisory':'wanderchat-travel-advisory-rag',
              'Wiki':'wanderchat-wiki-filtered-rag'}

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key = st.secrets['OPENAI_API_KEY'])

database = {'Reddit', 'SF Bay Area Activities', 'U.S. Travel Advisory'}

# Mapping of persona combinations to greetings
greetings_map = {
    frozenset(['Reddit', 'SF Bay Area Activities', 'U.S. Travel Advisory']):
        "Ready to navigate the worlds of Reddit, discover fun activities in the SF Bay Area, and stay updated with U.S. travel advisories? Which adventure shall we embark on first?",
    frozenset(['Reddit', 'SF Bay Area Activities']):
        "Are you looking to explore both the digital realms of Reddit and real-world activities in the SF Bay Area? Which one shall we start with today?",
    frozenset(['Reddit', 'U.S. Travel Advisory']):
        "Curious about the latest on Reddit or needing travel advice for a U.S. trip? Let me know what's on your mind first!",
    frozenset(['SF Bay Area Activities', 'U.S. Travel Advisory']):
        "Interested in discovering local events in the Bay Area or seeking travel tips? Which is more urgent for your plans?",
    frozenset(['Reddit']):
        "What topics are you currently interested in on Reddit? I can help you find the latest discussions on them!",
    frozenset(['SF Bay Area Activities']):
        "Looking for outdoor fun or a cozy indoor event in the Bay Area this weekend? Just ask, and I'll find you some options!",
    frozenset(['U.S. Travel Advisory']):
        "Planning a trip and need the latest travel advisories? Tell me your destination, and I'll provide the essential information!"
}


with st.sidebar:
    database = st.multiselect("Choose knowledge base:",index_dict.keys(),'Reddit')
    
    #define list to store retrievers
    retrievers = []
    for i in database:
        #connect to Pinecone vectore store
        vectorstore = Pinecone.from_existing_index(
            index_dict.get(i), embed_model.embed_query).as_retriever()
        retrievers.append(vectorstore)

    #uniform weights
    weights = 1/(len(retrievers))
    #define ensemble retriever from list of retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,weights=[weights for i in range(len(retrievers))])
    
    selected_keys = frozenset(database)
    greeting = greetings_map.get(selected_keys)
    
    st.session_state.message = [{"role":"assistant","content":"Hello, I am Wanderchat. " + greeting}]

system_prompt = '''Answer the question as if you are a travel agent and your goal is to provide excellent customer service and to provide
        personalized travel recommendations with reasonings based on their question. 
        Do not repeat yourself or include any links or HTML.'''
        


headers = {"Accept" : "application/json","Authorization": f"Bearer {st.secrets['hf_token']}","Content-Type": "application/json" }
def query(payload):
  response = requests.post(st.secrets['model_base_url'], headers=headers, json=payload)
  return response.json()

def find_match(input):
    result = ensemble_retriever.invoke(input)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(result)
    return ' '.join([d.page_content for d in reordered_docs])
    
if st.secrets['hf_token'] and st.secrets['model_base_url']:
    
    # chat = ChatOpenAI(model_name="tgi",openai_api_key=st.session_state['user_hf_token'],openai_api_base=st.session_state['model_base_url'],)
    # client = OpenAI(base_url=st.session_state['model_base_url'], api_key=st.session_state['user_hf_token'])
    # st.session_state['llm'] = chat
    memoryforchat=ConversationBufferMemory()
    # convo=ConversationChain(memory=memoryforchat,llm=chat,verbose=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memoryforchat.save_context({"input":message["human"]},{"outputs":message["AI"]})

    for message1 in st.session_state.message:
        with st.chat_message(message1["role"]):
            st.markdown(message1["content"])
            
    if prompt:=st.chat_input("Say Something"):
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.message.append({"role":"user","content":prompt})
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        context = find_match(prompt)
                        
                        augmented_prompt = f"""{system_prompt}
                        
                        DOCUMENT:
                        {context}

                        QUESTION:
                        {prompt}

                        INSTRUCTIONS:
                        Answer the users QUESTION using the DOCUMENT text above.
                        Keep your answer ground in the facts of the DOCUMENT.
                        If the DOCUMENT doesn’t contain the facts to answer the QUESTION say 'I do not know'
                        """
                        
                        print(augmented_prompt)
                        
                        input_len = len(augmented_prompt.split())
                        max_token_len = 1500-input_len-100 #100 buffer

                        start_time = time.time()
                        while True: #while loop for token
                            answer = query({'inputs': f"<s>[INST] {augmented_prompt} [/INST]",
                                        'parameters': {"max_new_tokens": max_token_len}})
                            if 'error' not in answer:
                                break  #exit the while loop if there is no error
                            max_token_len -= 100 #reduce by 100 in while loop
                            print(f"Failed to process prompt with token length: {max_token_len}")
                            if max_token_len <= 0:
                                break
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        answer = answer[0]['generated_text'].replace(f"<s>[INST] {augmented_prompt} [/INST]","")
                        answer = answer.replace(" . ",". ").strip()
                        answer = re.sub(r'<ANSWER>.*$', '', answer, flags=re.DOTALL) #RAFT specific
                        responce = re.sub(r'Final answer: .*$', '', answer, flags=re.DOTALL) #RAFT specific
                        
                        st.write(responce)
                        st.session_state.message.append({"role":"assistant","content":responce})
                        message={'human':prompt,"AI":responce}
                        st.session_state.chat_history.append(message)
                        st.session_state['model_responded'] = True
                    except:
                        responce = "Error generating response. Please ask again."
                        st.write(responce)
                        
    # if st.session_state['model_responded']:
    #     with st.sidebar.expander("Feedback"):
    #         if feedback:= st.text_input("",""):
    #             st.session_state['user_feedback'].append(feedback)
    #             st.toast("Thanks for your feedback!")
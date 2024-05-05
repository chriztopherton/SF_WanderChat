import streamlit as st
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


def retrieval_grader(question):
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
        
    llm = st.session_state['llm']

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt 
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    docs = st.session_state['retriever'].get_relevant_documents(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
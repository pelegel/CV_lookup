import json
from typing import List

from dotenv import load_dotenv
from langchain import hub
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from output_parser import CandidateSummary

load_dotenv()
from ingestion import ingest_files
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from output_parser import candidate_summary_parser, CandidateSummary
import random
import numpy as np


def format_answer(answer):
    import re
    if answer[0] != "[" and answer[-1] != "]":
        answer = "[" + answer + "]"
    answer = answer.replace("```", "")  # Remove triple quotes
    answer = answer.replace("json", "")  # Remove 'json' prefix
    answer = answer.replace("}{", "},\n{")  # Remove 'json' prefix
    answer = re.sub(r'}\s*\n\s*{', '},\n{', answer)   # Remove any excessive newlines between JSON objects
    return answer

def find_best_candidates(job: str) -> List[CandidateSummary]:
    llm = ChatOpenAI(model="gpt-4o", verbose=True, temperature=0)
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-3-large")

    # A prompt template for the task
    template = """Given the following job description and candidates' cvs, I want you to find the 3 candidates 
    that best fit the job according to their cv.
    Your answer should contain 1-3 candidate information, each of them at the format specified below. 
    Don't try to make up candidates, use context only!!! 
    job description: {job}
    
    {format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["job"],
        template=template,
        partial_variables={
            "format_instructions": candidate_summary_parser.get_format_instructions()
        },
    )
    # Format the summary prompt with the query
    formatted_prompt = summary_prompt_template.format(job=job)
    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),
    #                                          combine_docs_chain=combine_docs_chain)

    bm25_retriever = BM25Retriever.from_documents(vectorstore.similarity_search(job, k=3))

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore.as_retriever(search_kwargs={"k": 3}), bm25_retriever],
        weights=[0.5, 0.5]
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=ensemble_retriever, combine_docs_chain=combine_docs_chain)

    print("formatted_prompt: ", formatted_prompt)
    result = retrieval_chain.invoke({"input": formatted_prompt})
    print("result: ", result)
    return format_answer(result["answer"])





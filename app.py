import json
from typing import Set
import streamlit as st
import json as pyjson
from streamlit_chat import message
from main import find_best_candidates


st.header("CV Lookup Bot")
job = st.chat_input("Enter your job description here")

if (
        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if job:
    with st.spinner("Scanning all candidates to find your best matches..."):
        generated_response = find_best_candidates(job)
        st.session_state["user_prompt_history"].append(job)
        st.session_state["chat_answers_history"].append(generated_response)
        st.session_state["chat_history"].append(("human", job))
        st.session_state["chat_history"].append(("ai", generated_response))


if st.session_state["chat_answers_history"]:
    # The zip order indicates the order of the chat messages
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        with st.chat_message("human"):
            response = st.write(job)
        with st.chat_message("assistant"):
            response = st.write("The best fit candidates I have found are:")

        # Convert the string to a list of JSON objects
        json_list = json.loads(generated_response)
        lst = ["Candidate", "candidate"]
        print("json_list2: ", json_list[0])

        for i, json_obj in enumerate(json_list[0]):
            print("json_obj: ", json_obj)
            if isinstance(json_obj, dict) and json_obj.get("name") is not None:
                if not any(word in json_obj.get("name") for word in lst):
                    candidate_info = json.dumps(json_obj, indent=4)
                    st.code(candidate_info, language="json")
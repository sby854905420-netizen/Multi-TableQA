import os
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Tuple
from Llm.llm_loader import LLM
from datetime import datetime
from Utils.path_finder import find_all_paths
from Utils.prompt_registry import PromptRegistry
from jinja2 import Template
from Data.CISS.graph_construction import build_variable_dependency_graph
from Utils.sql_check import looks_like_query_sql
from Utils.embedding_tools import get_embeddings
from Utils.table_RAG import Table_RAG

def get_schema_embeddings(schema_info:pd.DataFrame, model:str) -> pd.DataFrame:
    schema_info = schema_info.fillna("")

    schema_info['embedding'] = schema_info.apply(
        lambda x: get_embeddings(f"name: {str(x['name'])}, full name: {str(x['fullname'])}, description: {str(x['description'])}" ,embedding_model=model),
        axis=1
    )
    return schema_info


def extract_caseid(text):
    matches = re.findall(r'\b\d{4,5}\b', text)
    for num_str in matches:
        case_id = int(num_str)
        if case_id > 2050:
            return case_id
    return None



def get_answers_TableRAG(df_qa_pairs:pd.DataFrame, schema_info:pd.DataFrame, Answer_llm:LLM):
    embedding_model = "text-embedding-3-small"
    schema_embeddings = get_schema_embeddings(schema_info,embedding_model)
    
    Answer_list = []
    information = []

    for idx, question in enumerate(list(df_qa_pairs['question'])):
        # indices = process_question_rag(question, question_decompose_llm, corpus_embeddings, embedding_model=embedding_model, threshold=0.8)
        # indices = list(indices)
        
        if idx < 1:
            relevant_df = Table_RAG(question, schema_embeddings,threshold=0.8)
            # print(relevant_df)

        case_id = extract_caseid(question)

        
    #     variables = []
    #     sheet_names = []
    #     for i in indices:
    #         # print(f"i: {i}")
    #         variables.append(df.iloc[i]['name'])
    #         sheet_names.append(df.iloc[i]['sheet'])

    #     find(case_id, variables, sheet_names)
        
    #     with open("data/logs/output.txt", "r", encoding="utf-8") as file:
    #         Output = file.read() 
    #     # with open("data/logs/temporary_indices.txt", "r", encoding="utf-8") as file:
    #     #         variables_description = file.read() 
    #     with open(log_file_path, 'w', encoding='utf-8') as log_file:
    #         log_file.write(f"Captured Output:\n{Output}\n\n")
    #         # PrptQ = PromptBuilder(base_content=variables_description)
    #         PrptQ = PromptBuilder(base_content='')
    #         PrptQ.add_text(Output)
    #         PrptQ.add_text(question+"\n"+"**Please read all the information provided above carefully and focus on question related information, answer the above question:**\n" + "\n")
    #         # PrptQ.add_text("When answering the question, if applicable, use the code mapping above to translate the letter codes into their human-readable crash configuration meanings. If the answer is not found in the data, please respond with 'I don't know'.")
    #         Prompt = PrptQ.get_prompt()
    #         log_file.write(f"Final Prompt to Answer_llm:\n{Prompt}\n\n")
    #         information.append(Output)
    #         answer = Answer_llm.query(Prompt)
    #         log_file.write(f"Generated Answer:\n{answer}\n\n")
    #         end_time = time.now()
    #         Time_list.append(end_time-start_time)
    #         time_this_case = end_time - start_time
    #         print  (f"Generated Answer:\n{answer}\n\n")
    #         Answer_list.append(answer)
    #     with open("data/logs/output.txt", "w", encoding="utf-8") as f:
    #         pass
        
    #     save_case_result_to_excel(question, case_id, variables, sheet_names, answer, GTAnswer,time_this_case)

    
                 
    # return Answer_list, Time_list, information
    return None

def main():
    # intialize the basic path
    basic_path = os.getcwd()
    # intialize all file pathes in the project
    keywords_description_file_path = "Data/CISS/DataBaseInfo/keywords_description.xlsx"
    total_question_answers_file_path = "Data/CISS/question_answer.xlsx"
    db_path = "Data/CISS/Database"
    prompts_dir = "Prompts"

    # extract the keywords descriptions of the columns from the table
    schema_info = pd.read_excel(os.path.join(basic_path,keywords_description_file_path))
    # extract all question answering pairs
    df_qa_pairs = pd.read_excel(os.path.join(basic_path,total_question_answers_file_path))

    Answer_llm = LLM(model_name = "gpt-5-mini-2025-08-07", provider= "openai")

    get_answers_TableRAG(df_qa_pairs,schema_info,Answer_llm)

    
if __name__ == "__main__":
    main()







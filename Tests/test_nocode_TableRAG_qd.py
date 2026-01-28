import re
import sys
import os
import pandas as pd
import openai
import numpy as np
from find_nocode import find
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/llm/')))
from llm_loader_HPC import LLM_HPC
from prompt_manager import PromptBuilder
from TableRAG_keywords_decompQuestion import process_question_rag
from TableRAG_keywords import Table_RAG
import time
from datetime import datetime as time
import re


def extract_caseid(text):
    matches = re.findall(r'\b\d{4,5}\b', text)
    for num_str in matches:
        case_id = int(num_str)
        if case_id > 2050:
            return case_id
    return None


def read_questions_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file.readlines() if line.strip()]
    return questions

def get_gpt4o_embeddings(texts, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=texts
    )
    return [np.array(item.embedding) for item in response.data]

def save_case_result_to_excel(
    question, case_id, variables, sheet_names, answer, ground_truth, time,
    excel_path="crash_case_results_qwen32b-70b.xlsx"
    # excel_path="crash_case_results_llama70b-70bnew2.xlsx"
):
  
    row = {
        "case_id": case_id,
        "question": question,
        "variables": str(variables),
        "sheet_names": str(sheet_names),
        "answer": str(answer),
        "ground_truth": str(ground_truth),
        "time": str(time),
    }


    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path, engine='openpyxl')

        df_final = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_final = pd.DataFrame([row])


    df_final.to_excel(excel_path, index=False)
    print(f"✅ Saved case {case_id} to {excel_path}")

def get_answers_TableRAG(questions_list, GTAnswer_list,question_decompose_llm):
    embedding_model="text-embedding-3-small"
    log_folder = "./data/logs/0507_logs"
    os.makedirs(log_folder, exist_ok=True)
    corpus = [str(row['key words']) for _, row in df.iterrows()]
    corpus_embeddings = get_gpt4o_embeddings(corpus, model=embedding_model)
    Answer_list = []
    Time_list=[]
    information = []
    for idx, question in enumerate(questions_list):
        # if idx<89:
        #     continue
        # if idx<430:
        #     continue
        if idx >630:
            break
        start_time = time.now()
        # indices = process_question_rag(question, question_decompose_llm, corpus_embeddings, embedding_model=embedding_model, threshold=0.8)
        # indices = list(indices)
        indices = Table_RAG(question, corpus_embeddings, embedding_model=embedding_model, threshold=0.8)
        indices = list(indices)
        
        log_file_path = os.path.join(log_folder, f"log_llm2andTableRAG_question_{idx+1}_{time.now().strftime('%Y%m%d_%H%M%S')}.txt")
        case_id = extract_caseid(question)
        GTAnswer = GTAnswer_list[idx]
        
        if case_id is None:
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"Question: {question}\n")
                log_file.write(f"Indices: {indices}\n")
                log_file.write("No case ID found in the question.\n")
            continue
        
        variables = []
        sheet_names = []
        for i in indices:
            # print(f"i: {i}")
            variables.append(df.iloc[i]['name'])
            sheet_names.append(df.iloc[i]['sheet'])

        find(case_id, variables, sheet_names)
        
        with open("data/logs/output.txt", "r", encoding="utf-8") as file:
            Output = file.read() 
        # with open("data/logs/temporary_indices.txt", "r", encoding="utf-8") as file:
        #         variables_description = file.read() 
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Captured Output:\n{Output}\n\n")
            # PrptQ = PromptBuilder(base_content=variables_description)
            PrptQ = PromptBuilder(base_content='')
            PrptQ.add_text(Output)
            PrptQ.add_text(question+"\n"+"**Please read all the information provided above carefully and focus on question related information, answer the above question:**\n" + "\n")
            # PrptQ.add_text("When answering the question, if applicable, use the code mapping above to translate the letter codes into their human-readable crash configuration meanings. If the answer is not found in the data, please respond with 'I don't know'.")
            Prompt = PrptQ.get_prompt()
            log_file.write(f"Final Prompt to Answer_llm:\n{Prompt}\n\n")
            information.append(Output)
            answer = Answer_llm.query(Prompt)
            log_file.write(f"Generated Answer:\n{answer}\n\n")
            end_time = time.now()
            Time_list.append(end_time-start_time)
            time_this_case = end_time - start_time
            print  (f"Generated Answer:\n{answer}\n\n")
            Answer_list.append(answer)
        with open("data/logs/output.txt", "w", encoding="utf-8") as f:
            pass
        
        save_case_result_to_excel(question, case_id, variables, sheet_names, answer, GTAnswer,time_this_case)

    
                 
    return Answer_list, Time_list, information

def read_questions_and_answers_from_file(file_path):

    df = pd.read_excel(file_path)

    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(" 'question' or 'answer' ")

    questions_list = df["question"].astype(str).tolist()
    answers_list = df["ground_truth"].astype(str).tolist()

    return questions_list, answers_list

df = pd.read_excel("notebooks/table_description_wordcloud_new.xlsx")
file_path ="projects/0516/QA_emnlp2/crash_case_results3.5.xlsx"
question_list, GTAnswer_list = read_questions_and_answers_from_file(file_path)

Answer_llm = LLM_HPC(model_name="llama3-70b", provider="transformers")

question_decompose_llm = LLM_HPC(model_name="deepseek-r1-Distill-Qwen-32B", provider="transformers")

Answer_list_TableRAG, Time_list_TableRAG, information = get_answers_TableRAG(question_list,GTAnswer_list,question_decompose_llm)
# Write answers to file

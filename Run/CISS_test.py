import os
import json
import sqlite3
import pandas as pd
from Llm.llm_loader import LLM
from Utils.path_finder import find_all_paths
from Utils.prompt_registry import PromptRegistry
from jinja2 import Template
from Data.CISS.graph_construction import build_variable_dependency_graph
from Utils.table_RAG import Table_RAG,get_embeddings
import networkx as nx
import sqlparse
from Utils.generate_text_answer import generate_answer


def get_schema_embeddings(schema_info:pd.DataFrame, model="text-embedding-3-small") -> pd.DataFrame:

    schema_info['embedding'] = schema_info.apply(
        lambda x: get_embeddings(f"column name: {str(x['column_name'])}, description: {str(x['description'])}" ,embedding_model=model),
        axis=1
    )
    return schema_info

def get_query_segments(question:str, prompts_registry:PromptRegistry, qd_LLM:LLM) -> str:
    # question_decomposition
    query_decomposition_prompt = prompts_registry.load("question_decomposition")
    system_msg = query_decomposition_prompt["system"]
    user_tpl = Template(query_decomposition_prompt["user_template"])
    user_msg = user_tpl.render({
        "question" : question
    })
    run_prompt = system_msg + "\n" + user_msg
    query_segments = qd_LLM.query(run_prompt)

    return query_segments

def parse_attributes_json(response: str, prompts_registry:PromptRegistry,json_LLM:LLM) -> list:
    text = response.strip()
    try:
        results = json.loads(text)
        return results['results']
    except Exception:
        pass
    
    extract_json_prompt = prompts_registry.load("extract_json")
    system_msg = extract_json_prompt["system"]
    user_tpl = Template(extract_json_prompt["user_template"])
    user_msg = user_tpl.render({
        "text" : text
    })
    run_prompt = system_msg + "\n" + user_msg

    json_text = json_LLM(run_prompt)
    
    try:
        results = json.loads(json_text.strip())
        return results['results']
    except Exception:
        return None
    
def find_reasoning_path(CISS_graph:nx.DiGraph, relevant_attributes:dict) -> list:
    condidates = []
    for sheet_name, column_name in relevant_attributes.values():
        start_node = (sheet_name, column_name)
        target_node = ('CASEID','center')
        avail_path = find_all_paths(CISS_graph,start_node,target_node)
        condidates = condidates + avail_path
    if condidates == []:
        print("No Path found!!!")
        return None
    
    condidates = [p for p in condidates if len(p) >= 3]
    condidates = [p[:-1] for p in condidates]
    return condidates

def format_paths(paths:list) -> str:
    lines = []
    for i, path in enumerate(paths, start=1):
        formatted_nodes = [
            f"{column}[{table}]" for column, table in path
        ]
        path_str = " → ".join(formatted_nodes)
        lines.append(f"Path {i}: {path_str}")
    return "\n".join(lines)

def run_sql_on_db(sql: str, conn: sqlite3.Connection):
    result_df = pd.read_sql_query(sql, conn)
    return result_df.to_dict(orient="records")

def parse_sql_text(sql_statement: str, prompts_registry:PromptRegistry,LLM:LLM) -> str:
    extract_json_prompt = prompts_registry.load("extract_sql_query")
    system_msg = extract_json_prompt["system"]
    user_tpl = Template(extract_json_prompt["user_template"])
    user_msg = user_tpl.render({
        "text" : sql_statement
    })
    run_prompt = system_msg + "\n" + user_msg
    sql_llm_text = LLM.query(run_prompt)

    return sql_llm_text

def match_result_to_text(match_result: dict) -> str:
    lines = []
    i = 1 
    for seg in match_result.keys():
        line = f"Match {i}: fragment = \"{seg}\" → column = \"{match_result[seg][0]}\" in table = \"{match_result[seg][1]}\""
        lines.append(line)
        i = i+1
    return "\n".join(lines)

def check_sql_syntax(sql:str,conn: sqlite3.Connection):
    try:
        parsed = sqlparse.parse(sql)
        try:
            conn.execute("EXPLAIN " + sql)
            return True
        except sqlite3.OperationalError:
            return False
    except SyntaxError:
        return False
    
def check_sql_valid(sql:str,conn: sqlite3.Connection, prompts_registry:PromptRegistry,LLM:LLM):
    if sql is None:
        return False, "No Valid SQL !!!!!"
    if check_sql_syntax(sql,conn):
        return True, run_sql_on_db(sql,conn)
    else:
        sql_fixed = parse_sql_text(sql,prompts_registry,LLM)
        if check_sql_syntax(sql_fixed,conn):
            return True, run_sql_on_db(sql,conn)
        else:
            return False, "No Valid SQL !!!!!"
        

def get_answers_TableRAG(query:str, schema_info_with_embedding:pd.DataFrame, prompts_registry:PromptRegistry, Answer_llm:LLM):
    
    CISS_graph = build_variable_dependency_graph()

    query_segments = get_query_segments(query, prompts_registry,Answer_llm)
    query_segments_list = parse_attributes_json(query_segments,prompts_registry,Answer_llm)

    if query_segments_list is None:
        print(f"[Skipped] No valid match_result for query: \n{query}")
        return None

    match_result = Table_RAG(query_segments_list, schema_info_with_embedding,threshold=0.8)
    if match_result == []:
        print("No relevant column retrived!!!")
        attribute_match_results = "No relevant column retrived!!!"
    else:
        Paths = find_reasoning_path(CISS_graph, match_result)
        Paths_text = format_paths(Paths)
        attribute_match_results = match_result_to_text(match_result)

    lines = []
    for _, r in schema_info_with_embedding.iterrows():
        kw = str(r['description']).strip().replace("\n", " ")
        code_discription = str(r['value_meaning']).strip().replace('\n',', ')
        lines.append(f"{r['sheet_name']} | {r['column_name']} | {r['type']} | {kw} | {code_discription}")
    schema_description_str = "\n".join(lines)

    nl2sql_prompt_tmplate = prompts_registry.load("nl2sql_with_table_CISS")
    system_tpl = Template(nl2sql_prompt_tmplate["system"])
    system_msg = system_tpl.render({
        "schema_description" : schema_description_str
    })
    user_tpl = Template(nl2sql_prompt_tmplate["user_template"])
    user_msg = user_tpl.render({
        "question" : query,
        "paths" : Paths_text,
        "alignment_information" : attribute_match_results
    })

    run_nl2sql_prompt = system_msg + "\n" + user_msg
    sql_statement_result = Answer_llm.query(run_nl2sql_prompt)
    
    return sql_statement_result    


def main():
    # intialize the basic path
    basic_path = os.getcwd()
    # intialize all file pathes in the project
    schema_description_file_path = "Data/CISS/DataBaseInfo/CISS_schema.xlsx"
    total_question_answers_file_path = "Data/CISS/question_answer.xlsx"
    db_path = "Data/CISS/SqlDatabase/CISS.sqlite"
    prompts_dir = "Prompts"

    # extract the keywords descriptions of the columns from the table
    schema_info = pd.read_excel(os.path.join(basic_path,schema_description_file_path))
    # extract all question answering pairs
    df_qa_pairs = pd.read_excel(os.path.join(basic_path,total_question_answers_file_path))

    Answer_llm = LLM(model_name = "gpt-5-mini-2025-08-07", provider= "openai")

    prompts_registry = PromptRegistry(prompt_dir=os.path.join(basic_path,prompts_dir))

    schema_info_with_embedding = get_schema_embeddings(schema_info)

    conn = sqlite3.connect(os.path.join(basic_path,db_path))


    for idx, question in enumerate(list(df_qa_pairs['question'])):
        if idx < 5:
            print("*****************************")
            print(f"Processing question {idx+1}")
            ground_truth = df_qa_pairs.iloc[idx]['ground_truth']
            sql_pred = get_answers_TableRAG(question,schema_info_with_embedding,prompts_registry,Answer_llm)

            ok, sql_answer_pred = check_sql_valid(sql_pred,conn,prompts_registry,Answer_llm)
            if ok:
                answer_pred = generate_answer(question,schema_info,sql_answer_pred,prompts_registry,Answer_llm)
                print(f"Query: {question}")
                print(f"Prediction: {answer_pred} vs Label : {ground_truth}")
            else:
                print(answer_pred)
    conn.close()
    
if __name__ == "__main__":
    main()






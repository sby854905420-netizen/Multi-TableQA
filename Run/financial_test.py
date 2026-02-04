import os
import json
import sqlite3
import pandas as pd
from typing import Tuple
from Llm.llm_loader import LLM
from datetime import datetime
from Utils.path_finder import find_all_paths
from collections import Counter
from Utils.prompt_registry import PromptRegistry
from jinja2 import Template
from Data.Financial.graph_construction import build_variable_dependency_graph
from Utils.sql_check import looks_like_query_sql


def get_question_answers(path:str, topics=["financial"], limit=10000) -> list:
    """
    Docstring for get_question_answers
    
    :param path: file path of the quesiton anwers json file
    :type path: str
    :param topics: list containing the required topics of the question anwering pairs
    :type path: list
    :param limit: maximum amounts of the question answering pairs
    :type path: int
    :return: list containing the question answering pairs with required topics
    :rtype: list
    """
    with open(path, "r") as f:
        total_qa_pairs = json.load(f)[:limit]

    related_questions = [
            item for item in total_qa_pairs if item.get("db_id") in topics
        ]
    print(f"Filtered questions: {len(related_questions)}")

    return related_questions

def classify_sql_type(sql: str) -> str:
    """
    Docstring for classify_sql_type
    
    :param sql: candidate sql statement
    :type sql: str
    :return: the type of the sql statement
    :rtype: str
    """
    sql = sql.lower()
    if "count(" in sql:
        return "Count"
    elif "group by" in sql or "distinct" in sql:
        return "List"
    elif any(agg in sql for agg in ["avg(", "sum(", "max(", "min("]):
        return "Numerical"
    elif "select" in sql:
        return "Select"
    else:
        return "Other"
    
def build_variable_alignment_prompt(schema_description:pd.DataFrame, question_item:pd.DataFrame, 
                                    prompts_registry:PromptRegistry) -> str:

    lines = []
    for _, r in schema_description.iterrows():
        kw = str(r['key_words']).strip().replace("\n", " ")
        lines.append(f"{r['sheet_name']} | {r['column_name']} | {kw}")
    schema_description_str = "\n".join(lines)

    messages = prompts_registry.render("schema_alignment_financial",
                               schema_description_str,
                               question_item["question"],
                               question_item["evidence"])
    
    final_prompt = prompts_registry.hash_prompt(messages)

    return final_prompt

def parse_attributes_json(response: str, prompts_registry:PromptRegistry,LLM:LLM) -> list:
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

    json_text = LLM(run_prompt)
    
    try:
        results = json.loads(json_text.strip())
        return results['results']
    except Exception:
        return None

def match_result_to_text(match_result: list) -> str:
    lines = []
    for i, entry in enumerate(match_result, 1):
        line = f"Match {i}: fragment = \"{entry['fragment']}\" → column = \"{entry['column_name']}\" in table = \"{entry['sheet_name']}\""
        lines.append(line)
    return "\n".join(lines)

def extract_variable_info(match_result: list) -> list:
    variables = [entry["column_name"] for entry in match_result]
    sheet_names = [entry["sheet_name"] for entry in match_result]
    return variables, sheet_names

def filter_all_paths(variables,sheet_names) -> list:
    Paths = []
    target_node = ("account_id", "account")
    G = build_variable_dependency_graph()
    for variable, sheet_name in zip(variables, sheet_names):
        start_node = (variable, sheet_name)
        Paths = Paths + find_all_paths(G,start_node, target_node)
    Paths = [p for p in Paths if p is not None and len(p) > 1]
    return Paths

def format_paths(paths:list) -> str:
    lines = []
    for i, path in enumerate(paths, start=1):
        formatted_nodes = [
            f"{column}[{table}]" for column, table in path
        ]
        path_str = " → ".join(formatted_nodes)
        lines.append(f"Path {i}: {path_str}")
    return "\n".join(lines)

def parse_sql_text(response: str, prompts_registry:PromptRegistry,LLM:LLM) -> Tuple[bool,str]:
    ok, reason, sql_text = looks_like_query_sql(response,require_from=False)
    if ok:
        return ok,sql_text
    else:
        print(reason)
        print("Try to extrac SQL query using LLM.")

        extract_json_prompt = prompts_registry.load("extract_sql_query")
        system_msg = extract_json_prompt["system"]
        user_tpl = Template(extract_json_prompt["user_template"])
        user_msg = user_tpl.render({
            "text" : response
        })
        run_prompt = system_msg + "\n" + user_msg
        sql_llm_text = LLM(run_prompt)

        ok, reason, sql_text = looks_like_query_sql(sql_llm_text,require_from=False)
        if ok:
            return ok,sql_llm_text
        else:
            return ok,sql_llm_text
        
def get_SQL_TableRAG(schema_description:pd.DataFrame, question_item:pd.DataFrame, 
                    prompts_registry:PromptRegistry, Answer_llm:LLM) -> Tuple[bool,str]:
    
    variable_alignment_prompt= build_variable_alignment_prompt(schema_description, question_item,prompts_registry)
    print(variable_alignment_prompt)
    variable_alignment_results = Answer_llm.query(variable_alignment_prompt)
    # print(f"Response: {variable_alignment_results}")
    match_result_list = parse_attributes_json(variable_alignment_results,prompts_registry,Answer_llm)
    if match_result_list is None:
        print(f"[Skipped] No valid match_result for question: {question_item['question']}")
        return None  
    attribute_match_results = match_result_to_text(match_result_list)
    variables, sheet_names = extract_variable_info(match_result_list)

    Paths = filter_all_paths(variables,sheet_names)
    if not Paths:
        print(f"[Skipped] No path found for question: {question_item['question']}")
        return None
    Paths_text = format_paths(Paths)

    nl2sql_prompt_tmplate = prompts_registry.load("nl2sql_with_table_financial")
    system_msg = nl2sql_prompt_tmplate["system"]
    user_tpl = Template(nl2sql_prompt_tmplate["user_template"])
    user_msg = user_tpl.render({
        "question" : question_item['question'],
        "paths" : Paths_text,
        "alignment_information" : attribute_match_results
    })
    run_nl2sql_prompt = system_msg + "\n" + user_msg

    sql_statement_result = Answer_llm.query(run_nl2sql_prompt)
    
    ok, final_pred_sql = parse_sql_text(sql_statement_result,prompts_registry,Answer_llm)

    return ok, final_pred_sql

def run_sql_on_db(sql: str, conn: sqlite3.Connection):
    result_df = pd.read_sql_query(sql, conn)
    return result_df.to_dict(orient="records")

def safe_parse(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return x
    return x

def normalize_row(row):
    return {k: v.strip().lower() if isinstance(v, str) else v for k, v in row.items()}

def results_equal(a, b):
    a = safe_parse(a)
    b = safe_parse(b)

    if not isinstance(a, list) or not isinstance(b, list):
        return a == b

    a_json = [json.dumps(normalize_row(row), sort_keys=True) for row in a]
    b_json = [json.dumps(normalize_row(row), sort_keys=True) for row in b]

    return Counter(a_json) == Counter(b_json)


def main():
    # intialize the basic path
    basic_path = os.getcwd()
    # intialize all file pathes in the project
    keywords_description_file_path = "Data/Financial/DataBaseInfo/keywords_description.xlsx"
    total_question_answers_file_path = "Data/Financial/question_answer.json"
    db_path = "Data/Financial/SqlDatabase/financial.sqlite"
    prompts_dir = "Prompts"

    # # Generate a timestamp for this run (used to avoid log overwrite)
    # time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # Construct a unique log directory for the current experiment
    # log_folder = os.path.join(basic_path,"Logs",f"financial_test_{time_tag}")
    # # Create the log directory if it does not already exist
    # os.makedirs(log_folder, exist_ok=True)

    # extract the keywords descriptions of the columns from the table
    schema_info = pd.read_excel(os.path.join(basic_path,keywords_description_file_path))

    # intialize the LLM model for answering the question
    Answer_llm = LLM(model_name = "gpt-5-mini-2025-08-07", provider= "openai")

    # extract all question answering pairs with respect to the financial
    financial_qa_pairs = get_question_answers(os.path.join(basic_path,total_question_answers_file_path),limit=10000)
    # print the distribution of the sql types among the financial question answering pairs 
    df_qa_pairs = pd.DataFrame(financial_qa_pairs)
    df_qa_pairs["type"] = df_qa_pairs["SQL"].apply(classify_sql_type)
    # print(df_qa_pairs["type"].value_counts())
    prompts_registry = PromptRegistry(prompt_dir=os.path.join(basic_path,prompts_dir))

    conn = sqlite3.connect(os.path.join(basic_path,db_path))
    grade = 0
    for idx, qa_pair in enumerate(financial_qa_pairs):
        if idx< 1:
            print("*****************************")
            print(f"Processing question {idx+1}")
            ok,sql_pred = get_SQL_TableRAG(schema_info,qa_pair, prompts_registry,Answer_llm)
            if ok:
                answer_pred = run_sql_on_db(sql_pred, conn)
                ground_truth = run_sql_on_db(qa_pair['SQL'], conn)
                correctness = results_equal(answer_pred, ground_truth)
                print(f"Prediction: {answer_pred} vs Label : {ground_truth}")

    conn.close()

    
if __name__ == "__main__":
    main()

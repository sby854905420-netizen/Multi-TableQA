import os
import json
import sqlite3
import pandas as pd
from Llm.llm_loader import LLM
from datetime import datetime
from Utils.path_finder import find_all_paths
from Utils.prompt_registry import PromptRegistry
from jinja2 import Template
from Data.Financial.graph_construction import build_variable_dependency_graph
import sqlparse
from Evaluate.compute_metrics import compute_metrics_BIRD
from tqdm import tqdm

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
    
def build_variable_alignment_prompt(schema_description:pd.DataFrame,
                                    question_item:pd.DataFrame, prompts_registry:PromptRegistry) -> str:
    lines = []
    for _, r in schema_description.iterrows():
        kw = str(r['key_words']).strip().replace("\n", " ")
        lines.append(f"{r['sheet_name']} | {r['column_name']} | {kw}")
    schema_description_str = "\n".join(lines)

    schema_alignment_prompt = prompts_registry.load("schema_alignment_financial")
    system_tpl = Template(schema_alignment_prompt["system"])
    system_msg = system_tpl.render({
        "schema_description" : schema_description_str
    })

    user_tpl = Template(schema_alignment_prompt["user_template"])
    user_msg = user_tpl.render({
        "question" : question_item['question'],
        "tips" : question_item['evidence']
    })

    run_prompt = system_msg + "\n\n" + schema_alignment_prompt["examples"] + "\n" + user_msg

    return run_prompt

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

    json_text = LLM.query(run_prompt)
    
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

def run_sql_on_db(sql: str, conn: sqlite3.Connection):
    result_df = pd.read_sql_query(sql, conn)
    return result_df.to_dict(orient="records")

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

def get_SQL(schema_description:pd.DataFrame, question_item:pd.DataFrame, 
                    prompts_registry:PromptRegistry, Answer_llm:LLM) -> str:
    
    variable_alignment_prompt= build_variable_alignment_prompt(schema_description, question_item,prompts_registry)
    variable_alignment_results = Answer_llm.query(variable_alignment_prompt)
    # print(f"Response: {variable_alignment_results}")
    match_result_list = parse_attributes_json(variable_alignment_results,prompts_registry,Answer_llm)
    if match_result_list is None:
        print(f"[Skipped] No valid match_result for question: \n{question_item['question']}")
        return None  
    attribute_match_results = match_result_to_text(match_result_list)
    variables, sheet_names = extract_variable_info(match_result_list)

    Paths = filter_all_paths(variables,sheet_names)
    if not Paths:
        # print(f"[Skipped] No path found for question: {question_item['question']}")
        Paths_text = Paths_text = f"No path found for question: {question_item['question']}"
    else:
        Paths_text = format_paths(Paths)

    lines = []
    for _, r in schema_description.iterrows():
        kw = str(r['key_words']).strip().replace("\n", " ")
        lines.append(f"{r['sheet_name']} | {r['column_name']} | {kw}")
    schema_description_str = "\n".join(lines)

    nl2sql_prompt_tmplate = prompts_registry.load("nl2sql_with_table_financial")
    system_tpl = Template(nl2sql_prompt_tmplate["system"])
    system_msg = system_tpl.render({
        "schema_description" : schema_description_str
    })
    user_tpl = Template(nl2sql_prompt_tmplate["user_template"])
    user_msg = user_tpl.render({
        "question" : question_item['question'],
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
    financial_schema_description_file_path = "Data/Financial/DataBaseInfo/financial_schema.xlsx"
    total_question_answers_file_path = "Data/Financial/question_answer.json"
    db_path = "Data/Financial/SqlDatabase/financial.sqlite"
    prompts_dir = "Prompts"

    # extract the keywords descriptions of the columns from the table
    schema_info = pd.read_excel(os.path.join(basic_path,financial_schema_description_file_path))

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
    
    preds = []
    labels = []
    for idx, qa_pair in tqdm(enumerate(financial_qa_pairs)):
        # print("*****************************")
        # print(f"Processing question {idx+1}")
        sql_pred = get_SQL(schema_info,qa_pair, prompts_registry,Answer_llm)

        ok, answer_pred = check_sql_valid(sql_pred,conn,prompts_registry,Answer_llm)
        if ok:
            preds.append(answer_pred)
        else:
            preds.append(False)
            print(answer_pred)
        ground_truth = run_sql_on_db(qa_pair['SQL'], conn)
        labels.append(ground_truth)
        # print(f"Prediction: {answer_pred} vs Label : {ground_truth}")

    exact_accuray,contain_accuray = compute_metrics_BIRD(preds,labels)    
    print(f"Exact Accuracy: {exact_accuray}, Contain Accuracy: {contain_accuray}")
    conn.close()

    
if __name__ == "__main__":
    main()

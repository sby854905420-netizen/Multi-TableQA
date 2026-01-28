import os
import json
import sqlite3
import pandas as pd
import re
import sys
import os
import pandas as pd
import openai
import numpy as np
from src.llm.llm_loader_HPC import LLM_HPC
from src.llm.llm_loader import LLM
from src.llm.prompt_manager import PromptBuilder
# from TableRAG_keywords_Olympics import process_question_rag
import time
from datetime import datetime as time
from graph_financial import find_all_paths

base_path = "QA_emnlp/data/dev"
json_path = os.path.join(base_path, "dev.json")
db_path = os.path.join(base_path, "financial/financial.sqlite")
output_path = os.path.join(base_path, "financial/financial.xlsx")

from collections import Counter
import json

# from collections import Counter
# import json

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


def prepare_questions(limit=10000):

    with open(json_path, "r") as f:
        train_data = json.load(f)[:limit]

    print(f"Total questions loaded: {len(train_data)}")


    related_questions = [
        item for item in train_data if item.get("db_id") == "financial"
    ]
    print(f"Filtered questions: {len(related_questions)}")

    return related_questions

def build_variable_alignment_prompt(schema_df: pd.DataFrame, question: str) -> str:

    instructions = """
    You are a helpful assistant for aligning natural language question phrases with database variables.

    You are given:
    1. A database schema table with sheet_name (table), column_name (field), and deacription.
    2. A natural language question.

    Your task is:
    - Split the question into key semantic fragments (e.g., date, bank, etc.).
    - For each fragment, return the most relevant column_name and its corresponding sheet_name based on the key_words.

    Tips:
    - You must only return matches that can be verified in the database contents.
    If the question or context contains the word "issuance" or related phrases (e.g., "insurance issuance", "insurance transaction"), then generate a condition on the column k_symbol with value "POJISTNE"

    Output format:
    ```json
    [
    {
        "fragment": "...",
        "column_name": "...",  #  only the field name and be an exact match from the schema (e.g., "date", not "date.account")
        "sheet_name": "..."  #  table name where the column belongs and must be an exact match from the schema(e.g., "account")
    }
    ] 
    ```
    
    ## example:
        questiion: "How many accounts who choose issuance after transaction are staying in East Bohemia region?"
        Output:   
        ```json
    [
        {
            "fragment": "accounts",
            "column_name": "account_id",
            "sheet_name": "account"
        },
        {
            "fragment": "POPLATEK PO OBRATU",
            "column_name": "frequency",
            "sheet_name": "account"
        },
        {
            "fragment": "East Bohemia region",
            "column_name": "A3",
            "sheet_name": "district"
        }
    ] 
    ```
    """
    # Format schema entries
    schema_text = "\n".join([
    f"- {row['sheet_name']}.{row['column_name']}: {str(row['key_words']).strip()}" 
    if pd.notnull(row['key_words']) and str(row['key_words']).strip()
    else f"- {row['sheet_name']}.{row['column_name']}"
    for _, row in schema_df.iterrows()
    ])

    # Combine into prompt
    prompt = f"""{instructions}

    ### Schema:
    {schema_text}

    ### Question:
    {question}

    ### Now extract and match:
    """
    return prompt

def build_variable_evidence_prompt(evidence, schema_df: pd.DataFrame, question: str) -> str:

    instructions = """
    You are a helpful assistant for aligning natural language question phrases with database variables.

    You are given:
    1. A database schema table with sheet_name (table), column_name (field), and deacription.
    2. A natural language question.
    3. evidence from the database.

    Your task is:
    - Split the question into key semantic fragments (e.g., date, bank, etc.).
    - For each fragment, return the most relevant column_name and its corresponding sheet_name based on the key_words based on evidence.

    Tips:
    - You must only return matches that can be verified in the database contents.
    If the question or context contains the word "issuance" or related phrases (e.g., "insurance issuance", "insurance transaction"), then generate a condition on the column k_symbol with value "POJISTNE"

    Output format:
    ```json
    [
    {
        "fragment": "...",
        "column_name": "...",  #  only the field name and be an exact match from the schema (e.g., "date", not "date.account")
        "sheet_name": "..."  #  table name where the column belongs and must be an exact match from the schema(e.g., "account")
    }
    ] 
    ```
    
    ## example:
        questiion: "How many accounts who choose issuance after transaction are staying in East Bohemia region?"
        Output:   
        ```json
    [
        {
            "fragment": "accounts",
            "column_name": "account_id",
            "sheet_name": "account"
        },
        {
            "fragment": "POPLATEK PO OBRATU",
            "column_name": "frequency",
            "sheet_name": "account"
        },
        {
            "fragment": "East Bohemia region",
            "column_name": "A3",
            "sheet_name": "district"
        }
    ] 
    ```
    """
    # Format schema entries
    schema_text = "\n".join([
    f"- {row['sheet_name']}.{row['column_name']}: {str(row['key_words']).strip()}" 
    if pd.notnull(row['key_words']) and str(row['key_words']).strip()
    else f"- {row['sheet_name']}.{row['column_name']}"
    for _, row in schema_df.iterrows()
    ])

    # Combine into prompt
    prompt = f"""{instructions}

    ### Schema:
    {schema_text}

    ### Question:
    {question}
    ### Evidence:
    {evidence}

    ### Now extract and match:
    """
    return prompt

def fix_sql_alias_with_spaces(sql: str) -> str:
    

    pattern = r'AS\s+((?!["\']).*?\s+[^,\n\r]*)'
    
    def replacer(match):
        alias = match.group(1).strip()
        if not alias.startswith('"') and not alias.endswith('"'):
            return f'AS "{alias}"'
        return match.group(0)  # already quoted

    fixed_sql = re.sub(pattern, replacer, sql)
    return fixed_sql


def run_sql_on_db(sql: str, conn: sqlite3.Connection):
    try:
        result_df = pd.read_sql_query(sql, conn)
        return result_df.to_dict(orient="records")
    except Exception as e:
        return f"[SQL Error] {e}"
    
def extract_sql(text):
 
    sql_block = re.search(r"```sql\s*(.+?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if sql_block:
        return sql_block.group(1).strip()

    sql_fallback = re.search(r"(SELECT\s.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL)
    if sql_fallback:
        return sql_fallback.group(1).strip()

    return None


def process_arrow_paths(paths):

    if len(paths) < 2:
        return paths
    split_paths = [[s.strip() for s in path.split(":", 1)[1].split("→")] for path in paths]


    min_len = min(len(p) for p in split_paths)
    common_suffix = []
    for i in range(1, min_len + 1):
        tokens = [p[-i] for p in split_paths]
        if all(t == tokens[0] for t in tokens):
            common_suffix.insert(0, tokens[0])
        else:
            break


    retained_suffix_token = common_suffix[0] if common_suffix else None

    result = []
    for i, steps in enumerate(split_paths):

        if retained_suffix_token and retained_suffix_token in steps:
            idx = steps.index(retained_suffix_token)
            trimmed = steps[:idx + 1]
        else:
            trimmed = steps
        result.append(f"Path {i+1}: {' → '.join(trimmed)}")

    return result

def normalize_in_string(text: str) -> str:
 
    if not isinstance(text, str):
        return text

    text = text.replace("Female", "F").replace("female", "F")
    text = text.replace("Male", "M").replace("male", "M")
    text = text.replace("weekly issuance", "POPLATEK TYDNE").replace("weekly", "POPLATEK TYDNE").replace("WEEKLY", "POPLATEK TYDNE")
    text = text.replace("issuance after transaction", "POPLATEK PO OBRATU")
    text = text.replace("monthly issuance", "POPLATEK MESICNE").replace("monthly", "POPLATEK MESICNE").replace("MONTHLY", "POPLATEK MESICNE")
    text = text.replace("class of credit card", "")
    text = text.replace("contract finished, no problems", "A")
    text = text.replace("ontract finished, loan not paid", "B")
    text = text.replace("running contract, OK so far", "C")
    text = text.replace("running contract, client in debt", "D")
    text = text.replace("insurance payment", "POJISTNE")
    text = text.replace("household payment", "SIPO")
    text = text.replace("leasing", "LEASING")
    text = text.replace("loan payment", "UVER")
    text = text.replace("payment for statement", "SLUZBY")
    text = text.replace("interest credited", "UROK")
    text = text.replace("sanction interest if negative balance", "SANKC. UROK")
    text = text.replace("old-age pension", "DUCHOD")
    text = text.replace("credit card withdrawal", "VYBER KARTOU")
    text = text.replace("credit in cash", "VKLAD")
    text = text.replace("collection from another bank", "PREVOD Z UCTU")
    text = text.replace("withdrawal in cash", "VYBER")
    text = text.replace("remittance to another bank", "PREVOD NA UCET")
    text = text.replace("household", "SIPO")
    text = text.replace("owner", "OWNER")
    text = text.replace("user", "USER")
    text = text.replace("disponent", "DISPONENT")
    text = text.replace("credit", "PRIJEM")
    text = text.replace("withdrawal", "VYDAJ")
    text = text.replace("active","A").replace("Active","A")


    
    return text


def process_arrow_paths_new(paths):

    if len(paths) < 2:
        return paths

    split_paths = [[s.strip() for s in path.split(":", 1)[1].split("→")] for path in paths]


    kept_paths = []
    for i, p1 in enumerate(split_paths):
        is_subpath = False
        for j, p2 in enumerate(split_paths):
            if i != j and len(p1) <= len(p2):
                
                for k in range(len(p2) - len(p1) + 1):
                    if p2[k:k+len(p1)] == p1:
                        is_subpath = True
                        break
            if is_subpath:
                break
        if not is_subpath:
            kept_paths.append((i, p1))

    if len(kept_paths) < 2:
   
        return [paths[i] for i, _ in kept_paths]


    path_steps = [p for _, p in kept_paths]
    min_len = min(len(p) for p in path_steps)
    common_suffix = []
    for i in range(1, min_len + 1):
        tokens = [p[-i] for p in path_steps]
        if all(t == tokens[0] for t in tokens):
            common_suffix.insert(0, tokens[0])
        else:
            break
    retained_suffix_token = common_suffix[0] if common_suffix else None


    result = []
    for path_idx, steps in kept_paths:
        if retained_suffix_token and retained_suffix_token in steps:
            idx = steps.index(retained_suffix_token)
            trimmed = steps[:idx + 1]
        else:
            trimmed = steps
        result.append(f"Path {len(result)+1}: {' → '.join(trimmed)}")

  
    return result

def match_result_to_text(match_result: list):
    lines = []
    for i, entry in enumerate(match_result, 1):
        line = f"Match {i}: fragment = \"{entry['fragment']}\" → column = \"{entry['column_name']}\" in table = \"{entry['sheet_name']}\""
        lines.append(line)
    return "\n".join(lines)

def extract_variable_info(match_result: list):
   
    variables = [entry["column_name"] for entry in match_result]
    sheet_names = [entry["sheet_name"] for entry in match_result]
    return variables, sheet_names

def extract_match_result_from_gpt_output(gpt_response: str):
 
    match = re.search(r"```json\s*(\[\s*{.*?}\s*])\s*```", gpt_response, re.DOTALL)
    if not match:
        print("[Warning] No valid ```json ...``` block found.")
        return None

    json_block = match.group(1)


    try:
        cleaned = json_block.replace("'", '"')  
        cleaned = re.sub(r"//.*", "", cleaned)  
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned) 
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[Error] JSON decode failed: {e}")
        print(f"[Raw JSON block]:\n{json_block}")
        return None
from collections import defaultdict
from typing import List

def filter_paths_with_common_endpoint(paths: List[str]) -> List[str]:

    endpoint_to_paths = defaultdict(list)

    for path in paths:
        if ": " in path:
            _, arrow_chain = path.split(": ", 1)
        else:
            arrow_chain = path
        endpoint = arrow_chain.strip().split("→")[-1].strip()
        endpoint_to_paths[endpoint].append(path)


    filtered_paths = []
    for endpoint, grouped in endpoint_to_paths.items():
        if len(grouped) >= 2:
            filtered_paths.extend(grouped)

    return filtered_paths


def quote_tokens_anywhere(sql: str) -> str:
    REPLACED_TOKENS = [
    "POPLATEK TYDNE", "POPLATEK MESICNE", "POPLATEK PO OBRATU",
    "VYBER KARTOU", "VKLAD", "PREVOD Z UCTU", "VYBER", "PREVOD NA UCET",
    "POJISTNE", "SIPO", "LEASING", "UVER", "SLUZBY",
    "UROK", "SANKC. UROK", "DUCHOD"
]
    for token in REPLACED_TOKENS:
     
        pattern = re.compile(rf'\b([a-zA-Z0-9_\-\.]*\b{re.escape(token)}\b[a-zA-Z0-9_\-\.]*)\b')
        
        def replacer(match):
            full_alias = match.group(1)
            if full_alias.startswith('"') and full_alias.endswith('"'):
                return full_alias 
            return f'"{full_alias}"'

        sql = pattern.sub(replacer, sql)
    return sql

def get_SQL_TableRAG(question, df, Answer_llm):
    
    Time_list=[]
    information = []
    start_time = time.now()

    df = pd.read_excel("QA_emnlp/data/dev/financial/financial_keywords.xlsx")
 
    prompt = build_variable_alignment_prompt(df, question)
    # print(f"Prompt: {prompt}")
    response = Answer_llm.query(prompt)
    # print(f"Response: {response}")
    match_result = extract_match_result_from_gpt_output(response)
    # print(f"Match Result: {match_result}")
    if match_result is None:
        print(f"[Skipped] No valid match_result for question: {question}")
        return None  
    # print(f"Match Result: {match_result}")
    information.append(match_result_to_text(match_result))
    variables, sheet_names = extract_variable_info(match_result)
    
    PrptQ = PromptBuilder(base_content='You are interacting with a Czech banking dataset stored in a relational database. The schema contains several related tables: account, client, district, loan, order, trans, card, and disp. Each table describes a different aspect of customer banking activity. Given the following reasoning paths and natural language question, generate the corresponding SQL query. Note: Some paths may be irrelevant to answering the question.\n ')
    PrptQ.add_text("""
Note: Some paths may be irrelevant to answering the question. Carefully identify and ignore unrelated paths to avoid introducing unnecessary joins or incorrect logic.

SQL Generation Guidelines:

1. **Fuzzy Text Matching**  
   For filtering only textual fields such as region or district names, use fuzzy matching with `LIKE '%...%'` instead of exact string equality.  
   ✅ Example: Use `A3 LIKE '%East Bohemia%'` rather than `A3 = 'East Bohemia region'`.

2. **Frequency Type Mapping**  
   Map natural language mentions of frequency to their corresponding values in `account.frequency`:  
   - "weekly" → `'POPLATEK TYDNE'`  
   - "monthly" → `'POPLATEK MESICNE'`  
   - "after transaction" → `'POPLATEK PO OBRATU'`

3. **Date Filtering**  
   When filtering by year, use SQLite syntax:  
   ✅ Use `STRFTIME('%Y', date) = '1997'` instead of `YEAR(date) = 1997`.

4. **Min/Max Selection**  
   When finding the minimum or maximum value (e.g., smallest loan), prefer:  
   ✅ `ORDER BY amount ASC/DESC LIMIT 1`  
   ❌ Avoid subqueries like `WHERE amount = (SELECT MIN(...))`.

5. **Field References**  
   - Use `trans.date` for transaction-related time filtering.  
   - Ensure the correct column is used for constraints mentioned in the question.

6. **Efficient Joins**  
   - Minimize joins to only what's necessary.  
   
7. **Aggregation Clarification**  
   Before using `COUNT(...)`, ensure whether the task is to count `clients`, `accounts`, or `transactions`, and select the appropriate key field.

8. **Consistency First**  
   When multiple filtering fields are involved (e.g., date + region + amount), ensure all conditions are consistently applied, with appropriate joins to relate the fields.
   
9. Do NOT invent or guess any column_name or sheet_name. Only use those provided in the paths.

10. If a table name is a SQL reserved keyword (e.g., order), you must wrap it in double quotes (e.g., "order") when writing the SQL query to avoid syntax errors.

11. When assigning column aliases, always wrap the alias in double quotes if it contains spaces, especially for values like "POPLATEK TYDNE". For example:
    ✅ AS "POPLATEK TYDNE_accounts_with_small_loan"
12. Do not include multiple fields in a single alias string.
13. When filtering on card.type, use exact known values such as 'junior', 'classic', or 'gold'. Do not use vague or invented terms like 'high' or 'premium'.
14. Only use DISTINCT when the question explicitly asks for unique values. If the question does not mention uniqueness (e.g., "different", "distinct", or "unique"), then do not use DISTINCT.    
    """)
# - Avoid joining through `disp` or `account` if the target field is already directly accessible (e.g., `client` joins to `district`).
    PrptQ.add_text(f"Question: {question}")
    Path = []
    for variable, sheet_name in zip(variables, sheet_names):
        Path.append(find_all_paths(variable, sheet_name))
        
    if Path is None:
        print(f"[Skipped] No path found for question: {question}")
        return None
    # print
    Path = format_paths_with_index(Path)
    # print(f"Path before: {Path}")
    Path = process_arrow_paths_new(Path)
    print(f"Path: {Path}")
    PrptQ.add_text("Please follow the path below:\n")
    PrptQ.add_text(" ".join(Path))
    PrptQ.add_text(f"information: {information}")
    PrptQ.add_text(f"""Example:
                    Paths:
                    Path 1: k_symbol[trans] → trans_id[trans]',
                    Path 2: A3[district] → district_id[district] → district_id[account] → account_id[account] → account_id[trans] → trans_id[trans], 
                    Path 3: A3[district] → district_id[district] → district_id[account] → account_id[account] → account_id[order] → order_id[order], 
                    Path 4: A3[district] → district_id[district] → district_id[account] → account_id[account] → account_id[loan] → loan_id[loan], 
                    Path 5: A3[district] → district_id[district] → district_id[account] → account_id[account] → account_id[disp] → disp_id[disp] → disp_id[card] → card_id[card] 

                    Question:  
                    How many accounts who choose issuance after transaction are staying in East Bohemia region?

                    ```sql
                    SELECT COUNT(DISTINCT account.account_id) AS issuance_lovers_in_east_bohemia
                    FROM account
                    JOIN trans ON account.account_id = trans.account_id
                    JOIN district ON account.district_id = district.district_id
                    WHERE district.A3 LIKE '%East Bohemia%'
                    AND account.frequency = 'POPLATEK PO OBRATU'
                    ```
                    
                    Paths:
                    Path 1: operation[trans] → trans_id[trans] → account_id[trans] → account_id[account] → account_id[disp] → disp_id[disp]',
                    Path 2: A2[district] → district_id[district] → district_id[client] → client_id[client] → client_id[disp] → disp_id[disp]', 
                    Path 3: A2[district] → district_id[district] → district_id[account] → account_id[account] → account_id[disp] → disp_id[disp]
                    Path 4: date[trans] → trans_id[trans] → account_id[trans] → account_id[account] → account_id[disp] → disp_id[disp
                    
                    Question:
                    Which are the top ten withdrawals (non-credit card) by district names for the month of January 1996?
                    
                    ```sql
                    SELECT district.A2 AS district_name,
                    trans.amount
                    FROM trans
                    JOIN account ON trans.account_id = account.account_id
                    JOIN disp ON account.account_id = disp.account_id
                    JOIN client ON disp.client_id = client.client_id
                    JOIN district ON client.district_id = district.district_id
                    WHERE trans.operation LIKE '%debit%' 
                    AND STRFTIME('%Y-%m', trans.date) = '1996-01'
                    ORDER BY trans.amount DESC
                    LIMIT 10;
                    ```
                    """)
    Prompt = PrptQ.get_prompt()
    answer = Answer_llm.query(Prompt)
    # print(f"Answer: {answer}")
    sql = extract_sql(answer)
    sql = normalize_in_string(sql)
    # sql = fix_sql_alias_with_spaces(sql)
    # sql = fix_unclosed_alias_quotes(sql)
    if sql is None:
        print("No SQL found in the answer.")
        return None
            
    return sql

def get_SQL(question,  df, Answer_llm):
    
    Time_list=[]
    information = []
    start_time = time.now()

    df = pd.read_excel("QA_emnlp/data/train/Olympics_keywords.xlsx")
 
    prompt = build_variable_alignment_prompt(df, question)
    # print(f"Prompt: {prompt}")
    response = Answer_llm.query(prompt)
    print(f"Response: {response}")
    match_result = extract_match_result_from_gpt_output(response)
    # print(f"Match Result: {match_result}")
    if match_result is None:
        print(f"[Skipped] No valid match_result for question: {question}")
        return None  
    # print(f"Match Result: {match_result}")
    information.append(match_result_to_text(match_result))
    variables, sheet_names = extract_variable_info(match_result)
    PrptQ = PromptBuilder(base_content='You are interacting with a Czech banking dataset stored in a relational database. The schema contains several related tables: account, client, district, loan, order, trans, card, and disp. Each table describes a different aspect of customer banking activity. Given the following natural language question, generate the corresponding SQL query. Note: Some paths may be irrelevant to answering the question.\n ')
    PrptQ.add_text("""
    
    # When generating SQL conditions for filtering textual fields such as region names, do not require exact string matches. Instead, use fuzzy matching via LIKE '%...%' to ensure partial matches are included (e.g., A3 LIKE '%East Bohemia%' instead of A3 = 'East Bohemia region').
    
    If the question mentions frequency types such as "weekly", "monthly", or "daily", map them to the exact values used in the account.frequency field:
    When filtering by year in a date field, replace YEAR(date) with STRFTIME('%Y', date) = '1997'.
    
    To find the minimum or maximum value (e.g., the smallest loan), prefer using ORDER BY ... LIMIT 1 instead of subqueries like WHERE amount = (SELECT MIN(...)) — it is simpler and more efficient.
    If the question involves trading date or transaction date, use the trans.date field.
    Prefer minimal necessary joins to improve execution efficiency and clarity.
    
    Determine whether the goal is to count clients or accounts (customers) before choosing your aggregation logic.
    Do NOT invent or guess any column_name or sheet_name. Only use those provided in the paths.
    """)
    PrptQ.add_text(f"Question: {question}")
   
    PrptQ.add_text(f"information: {information}")
    PrptQ.add_text(f"""Example: 
                   Question:  
                    How many accounts who choose issuance after transaction are staying in East Bohemia region?

                    ```sql
                    SELECT COUNT(DISTINCT account.account_id) AS issuance_lovers_in_east_bohemia
                    FROM account
                    JOIN trans ON account.account_id = trans.account_id
                    JOIN district ON account.district_id = district.district_id
                    WHERE district.A3 LIKE '%East Bohemia%'
                    AND account.frequency = 'POPLATEK PO OBRATU'
                    ```
                    
                    """)
    Prompt = PrptQ.get_prompt()
    answer = Answer_llm.query(Prompt)

    PrptQ.clear()
    if sql is None:
        print("No SQL found in the answer.")
        return None
            
    return sql
import re



def fix_unclosed_alias_quotes(sql: str) -> str:


    sql_keywords = ["FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "WHERE", "GROUP BY", "ORDER BY", "LIMIT"]

    lines = sql.split('\n')
    fixed_lines = []
    in_unclosed_alias = False
    alias_buffer = []

    for line in lines:
        stripped = line.strip()

        if in_unclosed_alias:
            alias_buffer.append(line)

            if any(kw in stripped.upper() for kw in sql_keywords):
     
                alias_content = ' '.join(alias_buffer)

                alias_before_kw = re.split(r'\b(FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT)\b', alias_content, flags=re.IGNORECASE)[0]
                if not alias_before_kw.strip().endswith('"'):
                    alias_before_kw += '"'
                fixed_lines.append(alias_before_kw)
                fixed_lines.append(line)
                in_unclosed_alias = False
                alias_buffer = []
            elif '"' in stripped:

                alias_line = ' '.join(alias_buffer)
                if alias_line.count('"') % 2 == 1:
                    alias_line += '"'
                fixed_lines.append(alias_line)
                in_unclosed_alias = False
                alias_buffer = []
            continue


        if re.search(r'\bAS\s+"[^"]*$', stripped, re.IGNORECASE):
            in_unclosed_alias = True
            alias_buffer = [line.rstrip()]
            continue

        fixed_lines.append(line)


    return '\n'.join(fixed_lines)

def format_paths_with_index(paths):
  
    result = []
    flat_paths = []


    for group in paths:
        if isinstance(group, list):
            for path in group:
                if isinstance(path, list) and all(isinstance(step, tuple) and len(step) == 2 for step in path):
                    flat_paths.append(path)


    for i, path in enumerate(flat_paths):
        try:
            arrow_path = " → ".join(f"{col}[{table}]" for col, table in path)
            result.append(f"Path {i+1}: {arrow_path}")
        except Exception as e:
            print(f"[Error] Formatting Path {i+1} failed: {e}")
            continue

    return result

def classify_sql_type(sql: str) -> str:
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

def main():
    df = pd.read_excel("/mimer/NOBACKUP/groups/naiss2025-22-321/projects/QA_emnlp/data/dev/financial/financial_keywords.xlsx")
    embedding_model="text-embedding-3-small"
    log_folder = "./data/train/logs/0507_logs"
    os.makedirs(log_folder, exist_ok=True)

    Answer_llm = LLM(model_name="gpt-4o", provider = "openai")

    questions = prepare_questions(limit=10000)
    qa_results = []
    df_q = pd.DataFrame(questions)

    df_q["type"] = df_q["SQL"].apply(classify_sql_type)


    type_counts = df_q["type"].value_counts()


    print(type_counts)
    quit()
    conn = sqlite3.connect(db_path)
    grade = 0
    for idx, item in enumerate(questions):
        if idx< 40:
            continue
        

        print(f"Processing question {idx+1}")
        question = item["question"]

        answer_all = []
        for time in range(10):
            sql = get_SQL_TableRAG(question,df, Answer_llm)
            answer = run_sql_on_db(sql, conn)
            sql_true = item["SQL"]
            ground_truth = run_sql_on_db(sql_true, conn)
            correctness = results_equal(answer, ground_truth)
            answer_all.append(answer)
            if correctness:
                print("✅ Answer is correct!")
                break
        
        print(f"SQL: {sql}")
        print(f"SQL True: {sql_true}")
        
        
        if correctness:

            grade += 1
            qa_results.append({
                "question": question,
                "sql": sql,
                "answer": answer,
                "ground_truth": ground_truth,
                "correctness": correctness,
                "sql_true": sql_true
                
            })
        else:
            print("❌ Answer is incorrect.")
            qa_results.append({
                "question": question,
                "sql": sql,
                "answer": answer_all,
                "ground_truth": ground_truth,
                "correctness": correctness,
                "sql_true": sql_true
                
            })
            qa = qa_results[idx-40]
            print(f" ****\n")
            # print(f"[Q{idx+1}] {qa['question']}")
            print(f" SQL: {qa['sql']}")
            print(f"SQL True: {qa['sql_true']}")
            
            # print(f" ****\n")

            print(f"GT: {qa['ground_truth']}\n")
            print(f" Ans: {qa['answer']}\n")

    conn.close()
    pd.DataFrame(qa_results).to_json("financial.json", indent=2)

    table_df = pd.DataFrame([
        {
            "Question": qa.get("question", ""),
            "Answer": str(qa.get("answer", "")),
            "Ground Truth": str(qa.get("ground_truth", "")),
            "SQL": qa.get("sql", ""),
            "SQL True": qa.get("sql_true", ""),
            "Correctness": qa.get("correctness", "")
        }
        for qa in qa_results if isinstance(qa, dict)
    ])
    table_df.to_csv("financial_result_table_gpt4o-pass10.csv", index=False)
    # table_df.to_csv("olympics.csv", index=False)
    print(f"Grade: {grade}/{len(questions)}")
if __name__ == "__main__":
    main()

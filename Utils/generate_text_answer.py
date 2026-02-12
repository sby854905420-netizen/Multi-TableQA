import sqlite3
from Utils.prompt_registry import PromptRegistry
from Llm.llm_loader import LLM
import pandas as pd
from jinja2 import Template



def generate_answer(query:str,schema_info:pd.DataFrame, sql_pred:list, 
                    prompts_registry:PromptRegistry,Answer_llm:LLM) -> str:

    lines = []
    for _, r in schema_info.iterrows():
        kw = str(r['description']).strip().replace("\n", " ")
        code_discription = str(r['value_meaning']).strip().replace('\n',', ')
        lines.append(f"{r['sheet_name']} | {r['column_name']} | {r['type']} | {kw} | {code_discription}")
    schema_description_str = "\n".join(lines)

    generate_answer_prompt_tmplate = prompts_registry.load("generate_answer_from_sql")
    system_msg = generate_answer_prompt_tmplate["system"]
    user_tpl = Template(generate_answer_prompt_tmplate["user_template"])
    user_msg = user_tpl.render({
        "question" : query,
        "schema_info" : schema_description_str,
        "sql_result_table" : sql_pred
    })

    generate_answer_prompt = system_msg + "\n" + user_msg
    predicted_text_answer = Answer_llm.query(generate_answer_prompt)

    return predicted_text_answer


from llm_loader import LLM
import ollama
from prompt_manager import PromptBuilder

def read_questions_from_file(file_path):
    
    
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file.readlines() if line.strip()]
    return questions



def parse_question_input(text):
    lines = text.strip().split("\n")
    conditions = []
    question = ""

    for line in lines:
        line = line.strip()
        if line.startswith("Condition"):
            _, value = line.split(":", 1)
            conditions.append(value.strip())
        elif line.startswith("Main Question:"):
            question = line.replace("Main Question:", "").strip()

    return conditions, question

def decompose_question(llm_model, question):
    file_path_p = "src/llm/prompt_qd.txt"
    with open(file_path_p, "r", encoding="utf-8") as file:
        base_prompt = file.read() 
    prompt_builder = PromptBuilder(base_prompt)
    prompt_builder.add_text(question)
    full_prompt = prompt_builder.get_prompt()
    # print(f"Full Prompt: {full_prompt}")
    answer = llm_model.query(full_prompt)

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

    prompt_builder.clear()
    conditions, main_question = parse_question_input(answer)
    return conditions, main_question





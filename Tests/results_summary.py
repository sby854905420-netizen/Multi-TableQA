import pandas as pd
import openai
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/llm/')))
from llm_loader import LLM
import json

# openai.api_key = "your-api-key"

def classify_hop_and_check_answer(question, answer, ground_truth):
    prompt = f"""
        You are a reasoning assistant for question classification and answer evaluation in a multi-table question answering (QA) system.
        Your task includes two parts:
        1. Classify the Question by Reasoning Hops
            Determine how many reasoning steps (hops) the question requires based on the number of distinct constraints or filters that must be applied across different attributes or entities.

            Guidelines:

            A 1-hop question only involves one constraint to locate the answer (e.g., "What is the color of vehicle X?").

            A 2-hop question requires two pieces of information to be jointly filtered (e.g., vehicle X where the driver is 32 years old).

            A 3-hop or more question involves three or more filters, entity linking steps, or sequential inferences.

            Example:

            Q: In case 16895, for the vehicle where the driver is 32 years old, what is the pre-crash critical event?
            → The hop_type is 2:

            hop 1: locate the vehicle in case 16895

            hop 2: filter for the vehicle with driver age = 32
            → Then query the critical event.

        Then, determine whether the model answer is correct. It is considered correct if it is semantically consistent with the ground-truth answer.
        2. Check Answer Containment
            Given the model's generated answer text, determine whether it contains the correct ground truth answer.

            Important:

            The generated text may include unrelated or extra information, which is acceptable.

            As long as the text semantically includes the correct answer (even within a longer paragraph), it is considered correct.

            Do not penalize over-generation unless the actual answer is missing or incorrect.

            Example:

            Q: What is the vehicle body type for the car in case 10010?
            Ground truth: "Sedan"
            Model output: "The car in case 10010 is a red Sedan with 4 doors."
            A: 
            {{
            "hop_type": 1
            "is_correct": true 
                    }}
            
        Only return your answer as JSON in following format:
        {{
            "hop_type": , // 1, 2 or 3
            "is_correct": true or false
        }}

        Question: {question}
        Model Answer: {answer}
        Ground Truth: {ground_truth}
        """
   
    output = ev_LLM.query(prompt)
   

    cleaned_output = output.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        result = json.loads(cleaned_output) 
    except json.JSONDecodeError:
        print(f"⚠️ JSON :\n{cleaned_output}")
        result = {}
    # print(result)
    return result

def evaluate_questions(input_path, output_path):
    """

    """
    df = pd.read_excel(input_path)
    results = []

    for inds, row in df.iterrows():
        # if inds>15:
        #     break
        question = str(row["question"])
        answer = str(row["answer"])
        ground_truth = str(row["ground_truth"])

        result = classify_hop_and_check_answer(question, answer, ground_truth)
        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "hop_type": result.get("hop_type", "error"),
            "is_correct": result.get("is_correct", False)
        })

    result_df = pd.DataFrame(results)
    result_df.to_excel(output_path, index=False)
    print(f"✔️ Results saved to: {output_path}")
    return result_df

def compute_hop_accuracy(result_df):
    """

    """
    result_df = result_df[result_df["hop_type"] != "error"]
    valid_df = result_df[result_df["hop_type"].isin([1, 2, 3])]
    grouped = valid_df.groupby("hop_type")["is_correct"].agg(total="count", correct="sum").reset_index()
    grouped["accuracy (%)"] = grouped["correct"] / grouped["total"] * 100
    return grouped

ev_LLM = LLM(model_name="gpt-4o", provider = "openai")

input_file = "QA_emnlp/crash_case_results4o.xlsx"
output_file = "reports/evaluated_results4o.xlsx"
df_result = evaluate_questions(input_file, output_file)
acc_by_hop = compute_hop_accuracy(df_result)
print(acc_by_hop)

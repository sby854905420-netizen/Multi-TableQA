import pandas as pd
import sys
import os
import openai
import torch.nn.functional as F
import warnings
from llm_loader import LLM
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/llm/')))
import re
import torch
from question_decompose import decompose_question

openai.api_key = ''
        
df = pd.read_excel("notebooks/table_description_wordcloud_new.xlsx")
import torch
import numpy as np
from torch.nn.functional import cosine_similarity

def get_gpt4o_embeddings(texts, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=texts
    )
    return [np.array(item.embedding) for item in response.data]

def Table_RAG(question, corpus_embeddings, embedding_model="text-embedding-3-small", threshold=0.9):
    # Step 1: Get embeddings for the corpus
    # corpus = [str(row['key words']) for _, row in df.iterrows()]
    # corpus_embeddings = get_gpt4o_embeddings(corpus, model=embedding_model)
    corpus_embeddings = torch.tensor(corpus_embeddings, dtype=torch.float)
    # print(f"Corpus embeddings: {corpus_embeddings}")
    # Step 2: Get embedding for the question
    query_embedding = get_gpt4o_embeddings([question], model=embedding_model)[0]
    query_embedding = torch.tensor(query_embedding, dtype=torch.float)
    # print(f"Query embedding: {query_embedding}")
    # Step 3: Compute cosine similarity
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), corpus_embeddings, dim=1)
    # print(f"Similarities: {similarities}")
    # Step 4: Normalize and filter based on threshold
    squared_similarities = similarities ** 2
    range_similarities = squared_similarities.max() - squared_similarities.min()

    if range_similarities == 0:
        normalized_similarities = torch.zeros_like(squared_similarities)
    else:
        normalized_similarities = (squared_similarities - squared_similarities.min()) / range_similarities
    above_threshold = (normalized_similarities > threshold).nonzero(as_tuple=True)[0]
    # print("Normalized similarities above threshold:")
    # for idx in above_threshold:
    #     print(f"Index: {idx.item()}, Score: {normalized_similarities[idx].item():.4f}")
    # Step 5: Select indices above threshold
    selected_indices = (normalized_similarities > threshold).nonzero(as_tuple=True)[0]

    if selected_indices.numel() == 0:
        return []  # No relevant rows

    # Step 6: Sort the selected indices by similarity
    sorted_indices = normalized_similarities[selected_indices].argsort(descending=True)
    top_indices = selected_indices[sorted_indices].cpu().numpy()

    return top_indices


def read_questions_from_file(file_path):
    
    
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file.readlines() if line.strip()]
    return questions


def process_question_rag(question, Answer_llm, corpus_embeddings, embedding_model, threshold=0.85):
    """
    Decompose and run attribute-level RAG on the target question.
    
    Args:
        questions_list (list): List of questions.
        target_index (int): Index of the question to process.
        Answer_llm (callable): Function to decompose question into conditions and main question.
        corpus_embeddings (np.array): Precomputed corpus embeddings for RAG.
        embedding_model (object): Embedding model to use for query embedding.
        df (pd.DataFrame): DataFrame containing variable descriptions (columns: 'name', 'description').
        threshold (float): Similarity threshold for retrieval (default = 0.8).
    
    Returns:
        None (prints output directly)
    """
    # print(f"Question: {question}")
    conditions, main_question = decompose_question(Answer_llm, question)
    all_retrieved_indices = set()

    # Run RAG on each condition
    for condition in conditions:
        cond_indices = Table_RAG(condition, corpus_embeddings, embedding_model=embedding_model, threshold=threshold)
        all_retrieved_indices.update(cond_indices)

    # Run RAG on main question
    main_indices = Table_RAG(main_question, corpus_embeddings, embedding_model=embedding_model, threshold=threshold)
    all_retrieved_indices.update(main_indices)

    # Print retrieved variable names and descriptions
    # for i in sorted(all_retrieved_indices):
    #     print(f"- {df.iloc[i]['name']}: {df.iloc[i]['description']}  \n")
    # print("\n\n")
    return all_retrieved_indices


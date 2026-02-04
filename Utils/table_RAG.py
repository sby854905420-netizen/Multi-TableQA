import pandas as pd
import torch.nn.functional as F
from Utils.embedding_tools import get_embeddings
import torch
import numpy as np

def Table_RAG(question:str, corpus_embeddings: pd.DataFrame, embedding_model="text-embedding-3-small", threshold=0.8) -> pd.DataFrame:
    # Step 1: Get embedding for the question
    query_embedding = get_embeddings(question,embedding_model=embedding_model)
    # Step 2: Compute cosine similarity
    corpus_embeddings['cos_value'] = corpus_embeddings['embedding'].apply(
        lambda x: F.cosine_similarity(query_embedding, x, dim=0)
    )
    # print(f"Similarities: {similarities}")
    # Step 3: Normalize and filter based on threshold
    corpus_embeddings['squared_cos_value'] = corpus_embeddings['cos_value'] ** 2
    range_similarities = corpus_embeddings['squared_cos_value'].max() - corpus_embeddings['squared_cos_value'].min()

    if range_similarities == 0:
        corpus_embeddings['normalized_similarities'] = 0
        # normalized_similarities = torch.zeros_like(corpus_embeddings['squared_cos_value'].sample(n=1).iloc[0])
    else:
        corpus_embeddings['normalized_similarities'] = (torch.tensor(corpus_embeddings['squared_cos_value'],dtype=torch.float) - 
                                                        corpus_embeddings['squared_cos_value'].min()) / range_similarities
                                                        
    # Step 4: find all data where normailized similaritie is larger than threshold
    above_threshold_index = corpus_embeddings['normalized_similarities'] > threshold

    if len(above_threshold_index) == 0:
            return []
    else:
    # Step 5: Sort the selected indices by similarity
        sorted_df = corpus_embeddings.loc[above_threshold_index].sort_values(
            by='normalized_similarities',
            ascending=False
        )

    return sorted_df
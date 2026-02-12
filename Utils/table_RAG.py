import pandas as pd
import torch.nn.functional as F
import torch
from openai import OpenAI
import os


def get_embeddings(text:str, embedding_model="text-embedding-3-small") -> torch.Tensor:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    
    text = text.strip().replace('\n',' ')
    embedding = client.embeddings.create(input = [text], model=embedding_model).data[0].embedding
    return torch.tensor(embedding, dtype=torch.float)



def Table_RAG(segments:list, schema_info: pd.DataFrame, embedding_model="text-embedding-3-small", threshold=0.8) -> dict:
    # Step 1: Get embedding for the question
    segments = [v for d in segments for v in d.values()]
    query_segments_embedding = torch.stack([get_embeddings(s) for s in segments])
    # Step 2: Get embedding for the schema
    schema_embeddings = torch.stack(list(schema_info['embedding']))
    # Step 3: square and normalize the cosine value
    # vectors are unit vectors
    cos_sim_matrix = query_segments_embedding @ schema_embeddings.T
    # element-wise square
    sq = cos_sim_matrix ** 2
    # row-wise Min-Max normalization
    row_min = sq.min(dim=1, keepdim=True).values
    row_max = sq.max(dim=1, keepdim=True).values 
    normalized = (sq - row_min) / (row_max - row_min + 1e-12)
    # # Step 4: find all data where normailized similaritie is larger than threshold
    indices = (normalized > threshold).nonzero(as_tuple=False).tolist()
    if indices == []:
        return []
    
    dict = {}
    for index in indices:
        dict[segments[index[0]]] = (schema_info.iloc[index[1]]['column_name'],schema_info.iloc[index[1]]['sheet_name'])

    return dict




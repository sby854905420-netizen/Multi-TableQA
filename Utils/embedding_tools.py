from openai import OpenAI
import torch


def get_embeddings(text:str, embedding_model="text-embedding-3-small") -> torch.Tensor:
    client = OpenAI()    
    text = text.strip().replace('\n',' ')
    embedding = client.embeddings.create(input = [text], model=embedding_model).data[0].embedding
    
    return torch.tensor(embedding, dtype=torch.float)
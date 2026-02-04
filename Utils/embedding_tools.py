from openai import OpenAI
import torch


def get_embeddings(text:str, embedding_model="text-embedding-3-small") -> torch.Tensor:
    client = OpenAI(api_key='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')    
    text = text.strip().replace('\n',' ')
    embedding = client.embeddings.create(input = [text], model=embedding_model).data[0].embedding
    
    return torch.tensor(embedding, dtype=torch.float)
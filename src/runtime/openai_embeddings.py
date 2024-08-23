#! python3
# -*- encoding: utf-8 -*-
'''
@File    : openai_embeddings.py
@Time    : 2024/08/22 14:14:11
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
'''


from openai import Client

from base.types import Documents, Embeddings
from base.embedding_function import EmbeddingFunction


class OpenAIEmbeddingsFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_key: str, base_url: str, model_name: str, **kwargs):
        try:
            self.kwargs = kwargs
            self.model_name = model_name
            self.client = Client(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenAI Client: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        input = [t.replace("\n", " ") for t in input]
        try:
            response = self.client.embeddings.create(
                model = self.model_name,
                input = input,
            )
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI Embeddings API: {e}")
        
        return [data.embedding for data in response.data]
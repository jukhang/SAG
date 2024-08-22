#! python3
# -*- encoding: utf-8 -*-
'''
@File    : vector_store.py
@Time    : 2024/08/22 13:58:51
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
'''


from typing import Dict
from abc import ABC, abstractmethod
from runtime.openai_embeddings import OpenAIEmbeddingsFunction


class VectorStore(ABC):
    '''
    Vector Base Class
    '''
    def __init__(self, config: Dict = {}) -> None:
        self.config = config
        self.dialect = self.config.get('dialect', 'SQL')
        if "embedding_fucntion" in self.config:
            self.embedding_function = self.config.get('embedding_function')
        else:
            self.embedding_function = OpenAIEmbeddingsFunction(
                api_key=self.config.get('api_key'),
                base_url=self.config.get('base_url'),
                model_name=self.config.get('embedding_model')
            )
        
        if "embedding_dim" in self.config:
            self.embedding_dim = self.config.get('embedding_dim')
        else:
            self.embedding_dim = 1024
        self.top_k = self.config.get('top_k', 3)
        self._create_collections()

    def _create_collections(self):
        self._create_sql_collections("sagsql")
        self._create_doc_collections("sagdoc")
        self._create_ddl_collections("sagddl")

    def _create_sql_collections(self, collection_name: str):
        pass

    def _create_doc_collections(self, collection_name: str):
        pass

    def _create_ddl_collections(self, collection_name: str):
        pass
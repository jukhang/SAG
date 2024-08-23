#! python3
# -*- encoding: utf-8 -*-
'''
@File    : milvus_vector.py
@Time    : 2024/08/22 15:14:54
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
'''


from typing import Dict
from pymilvus import DataType, MilvusClient, model


from base.vector_store import VectorStore

DEFAULT_MILVUS_URI = "http://localhost:19530"

class Milvus_VecotrStore(VectorStore):
    def __init__(self, config: Dict):
        super().__init__(config=config)

        if "milvus_client" in config:
            self.milvus_client = config["milvus_client"]
        else:
            self.milvus_client = MilvusClient(uri=DEFAULT_MILVUS_URI)

        self._create_collections()

    def _create_sql_collections(self, name: str):
        if not self.milvus_client.has_collection(name):
            sagsql_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_fields=False,
            )
            sagsql_schema.add_field(field_name="id",datatype=DataType.VARCHAR, max_length=255, is_primary=True)
            sagsql_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            sagsql_schema.add_field(field_name="sql", datatype=DataType.VARCHAR, max_length=65535)
            sagsql_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)

            index_params = self.milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector",
                index_type="AUTOINDEX",
                metric_type="L2",
            )

            self.milvus_client.create_collection(
                collection_name=name,
                schema=sagsql_schema,
                index_params=index_params,
                consistency_level="Strong",
            )

    def _create_doc_collections(self, name: str):
        if not self.milvus_client.has_collection(collection_name=name):
            vannadoc_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            vannadoc_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            vannadoc_schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)
            vannadoc_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)

            vannadoc_index_params = self.milvus_client.prepare_index_params()
            vannadoc_index_params.add_index(
                field_name="vector",
                index_name="vector",
                index_type="AUTOINDEX",
                metric_type="L2",
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=vannadoc_schema,
                index_params=vannadoc_index_params,
                consistency_level="Strong"
            )
        

    def _create_ddl_collections(self, name: str):
        if not self.milvus_client.has_collection(collection_name=name):
            vannaddl_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            vannaddl_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            vannaddl_schema.add_field(field_name="ddl", datatype=DataType.VARCHAR, max_length=65535)
            vannaddl_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)

            vannaddl_index_params = self.milvus_client.prepare_index_params()
            vannaddl_index_params.add_index(
                field_name="vector",
                index_name="vector",
                index_type="AUTOINDEX",
                metric_type="L2",
            )
            self.milvus_client.create_collection(
                collection_name=name,
                schema=vannaddl_schema,
                index_params=vannaddl_index_params,
                consistency_level="Strong"
            )
 
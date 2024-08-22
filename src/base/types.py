#! python3
# -*- encoding: utf-8 -*-
'''
@File    : types.py
@Time    : 2024/08/22 13:32:28
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
'''



import numpy as np
from numpy.typing import NDArray
from typing import Union, List,TypeVar, Sequence


# Documents
Document = str
Documents = List[Document]

# Images
ImageDType = Union[np.uint, np.int_, np.float_]  # type: ignore[name-defined]
Image = NDArray[ImageDType]
Images = List[Image]

# Embeddings
Vector = Union[Sequence[float], Sequence[int]]
Embedding = Vector
Embeddings = List[Embedding]

Embeddable = Union[Documents, Images]

T = TypeVar("T")
D = TypeVar("D", bound=Embeddable, contravariant=True)

# OneOrMany
OneOrMany = Union[T, List[T]]

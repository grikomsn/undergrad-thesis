from typing import Union

from pandas import DataFrame, Series
from pandas.api.extensions import ExtensionArray
from pandas.io.parsers import TextFileReader

UnifiedDataFrame = Union[DataFrame, ExtensionArray, Series, TextFileReader]

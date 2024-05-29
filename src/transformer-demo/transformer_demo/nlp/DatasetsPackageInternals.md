# Datasets package internals

## Interface for DatasetDict
```python
class DatasetDict(dict):
    @property
    def data(self) -> Dict[str, Table]:
        """The Apache Arrow tables backing each split.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.data
        ```
        """
        
      @property
    def cache_files(self) -> Dict[str, Dict]:
        """The cache files containing the Apache Arrow table backing each split.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.cache_files
        {'test': [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-test.arrow'}],
         'train': [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-train.arrow'}],
         'validation': [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-validation.arrow'}]}
        ```
        """

    @property
    def num_columns(self) -> Dict[str, int]:
        """Number of columns in each split of the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.num_columns
        {'test': 2, 'train': 2, 'validation': 2}
        ```
        """

    @property
    def num_rows(self) -> Dict[str, int]:
        """Number of rows in each split of the dataset (same as :func:`datasets.Dataset.__len__`).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.num_rows
        {'test': 1066, 'train': 8530, 'validation': 1066}
        ```
        """

    @property
    def column_names(self) -> Dict[str, List[str]]:
        """Names of the columns in each split of the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.column_names
        {'test': ['text', 'label'],
         'train': ['text', 'label'],
         'validation': ['text', 'label']}
        ```
        """

    @property
    def shape(self) -> Dict[str, Tuple[int]]:
        """Shape of each split of the dataset (number of columns, number of rows).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.shape
        {'test': (1066, 2), 'train': (8530, 2), 'validation': (1066, 2)}
        ```
        """

    def flatten(self, max_depth=16) -> "DatasetDict":
        """Flatten the Apache Arrow Table of each split (nested features are flatten).
        Each column with a struct type is flattened into one column per struct field.
        Other columns are left unchanged.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("squad")
        >>> ds["train"].features
        {'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
         'context': Value(dtype='string', id=None),
         'id': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None),
         'title': Value(dtype='string', id=None)}
        >>> ds.flatten()
        DatasetDict({
            train: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
                num_rows: 87599
            })
            validation: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
                num_rows: 10570
            })
        })
        ```
        """
```

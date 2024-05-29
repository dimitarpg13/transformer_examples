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
        ...

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
        ...
```

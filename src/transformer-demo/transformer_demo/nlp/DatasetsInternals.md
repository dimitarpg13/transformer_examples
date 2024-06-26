# Datasets internals

## Introduction and docs on huggingface

https://huggingface.co/docs/datasets/en/about_arrow

https://huggingface.co/docs/datasets/en/about_cache

https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable

https://huggingface.co/docs/datasets/en/about_dataset_features

https://huggingface.co/docs/datasets/en/about_dataset_load

https://huggingface.co/docs/datasets/en/about_map_batch

https://huggingface.co/docs/datasets/en/about_metrics


## Interface for Features

Summary of the members:

Docstrings and examples

```python
class Features(dict):
    """A special dictionary that defines the internal structure of a dataset.

    Instantiated with a dictionary of type `dict[str, FieldType]`, where keys are the desired column names,
    and values are the type of that column.

    `FieldType` can be one of the following:
        - a [`~datasets.Value`] feature specifies a single typed value, e.g. `int64` or `string`.
        - a [`~datasets.ClassLabel`] feature specifies a field with a predefined set of classes which can have labels
          associated to them and will be stored as integers in the dataset.
        - a python `dict` which specifies that the field is a nested field containing a mapping of sub-fields to sub-fields
          features. It's possible to have nested fields of nested fields in an arbitrary manner.
        - a python `list` or a [`~datasets.Sequence`] specifies that the field contains a list of objects. The python
          `list` or [`~datasets.Sequence`] should be provided with a single sub-feature as an example of the feature
          type hosted in this list.

          <Tip>

           A [`~datasets.Sequence`] with a internal dictionary feature will be automatically converted into a dictionary of
           lists. This behavior is implemented to have a compatilbity layer with the TensorFlow Datasets library but may be
           un-wanted in some cases. If you don't want this behavior, you can use a python `list` instead of the
           [`~datasets.Sequence`].

          </Tip>

        - a [`Array2D`], [`Array3D`], [`Array4D`] or [`Array5D`] feature for multidimensional arrays.
        - an [`Audio`] feature to store the absolute path to an audio file or a dictionary with the relative path
          to an audio file ("path" key) and its bytes content ("bytes" key). This feature extracts the audio data.
        - an [`Image`] feature to store the absolute path to an image file, an `np.ndarray` object, a `PIL.Image.Image` object
          or a dictionary with the relative path to an image file ("path" key) and its bytes content ("bytes" key). This feature extracts the image data.
        - [`~datasets.Translation`] and [`~datasets.TranslationVariableLanguages`], the two features specific to Machine Translation.
    """

```


## Interface for DatasetDict
Summary of the members:
```python
class DatasetDict(dict):
    @property
    def data(self) -> Dict[str, Table]
        """The Apache Arrow tables backing each split."""

    
    
```

Docstrings and examples:
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

    def unique(self, column: str) -> Dict[str, List]:
        """Return a list of the unique elements in a column for each split.

        This is implemented in the low-level backend and as such, very fast.

        Args:
            column (`str`):
                column name (list all the column names with [`~datasets.Dataset.column_names`])

        Returns:
            Dict[`str`, `list`]: Dictionary of unique elements in the given column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.unique("label")
        {'test': [1, 0], 'train': [1, 0], 'validation': [1, 0]}
        ```
        """

        def cleanup_cache_files(self) -> Dict[str, int]:
        """Clean up all cache files in the dataset cache directory, excepted the currently used cache file if there is one.
        Be careful when running this command that no other process is currently using other cache files.

        Return:
            `Dict` with the number of removed files for each split

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.cleanup_cache_files()
        {'test': 0, 'train': 0, 'validation': 0}
        ```
        """

        def cast(self, features: Features) -> "DatasetDict":
        """
        Cast the dataset to a new set of features.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can also remove a column using [`Dataset.map`] with `feature` but `cast`
        is in-place (doesn't copy the data to a new dataset) and is thus faster.

        Args:
            features ([`Features`]):
                New features to cast the dataset to.
                The name and order of the fields in the features must match the current column names.
                The type of the data must also be convertible from one type to the other.
                For non-trivial conversion, e.g. `string` <-> `ClassLabel` you should use [`~Dataset.map`] to update the Dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> new_features = ds["train"].features.copy()
        >>> new_features['label'] = ClassLabel(names=['bad', 'good'])
        >>> new_features['text'] = Value('large_string')
        >>> ds = ds.cast(new_features)
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='large_string', id=None)}
        ```
        """

        def cast_column(self, column: str, feature) -> "DatasetDict":
        """Cast column to feature for decoding.

        Args:
            column (`str`):
                Column name.
            feature ([`Feature`]):
                Target feature.

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> ds = ds.cast_column('label', ClassLabel(names=['bad', 'good']))
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='string', id=None)}
        ```
        """

        def remove_columns(self, column_names: Union[str, List[str]]) -> "DatasetDict":
        """
        Remove one or several column(s) from each split in the dataset
        and the features associated to the column(s).

        The transformation is applied to all the splits of the dataset dictionary.

        You can also remove a column using [`Dataset.map`] with `remove_columns` but the present method
        is in-place (doesn't copy the data to a new dataset) and is thus faster.

        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to remove.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.remove_columns("label")
        DatasetDict({
            train: Dataset({
                features: ['text'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text'],
                num_rows: 1066
            })
        })
        ```
        """

        def rename_column(self, original_column_name: str, new_column_name: str) -> "DatasetDict":
        """
        Rename a column in the dataset and move the features associated to the original column under the new column name.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can also rename a column using [`~Dataset.map`] with `remove_columns` but the present method:
            - takes care of moving the original features under the new column name.
            - doesn't copy the data to a new dataset and is thus much faster.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.rename_column("label", "label_new")
        DatasetDict({
            train: Dataset({
                features: ['text', 'label_new'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text', 'label_new'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text', 'label_new'],
                num_rows: 1066
            })
        })
        ```
        """

        def rename_columns(self, column_mapping: Dict[str, str]) -> "DatasetDict":
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            column_mapping (`Dict[str, str]`):
                A mapping of columns to rename to their new names.

        Returns:
            [`DatasetDict`]: A copy of the dataset with renamed columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.rename_columns({'text': 'text_new', 'label': 'label_new'})
        DatasetDict({
            train: Dataset({
                features: ['text_new', 'label_new'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text_new', 'label_new'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text_new', 'label_new'],
                num_rows: 1066
            })
        })
        ```
        """

        def select_columns(self, column_names: Union[str, List[str]]) -> "DatasetDict":
        """Select one or several column(s) from each split in the dataset and
        the features associated to the column(s).

        The transformation is applied to all the splits of the dataset
        dictionary.

        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to keep.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.select_columns("text")
        DatasetDict({
            train: Dataset({
                features: ['text'],
                num_rows: 8530
            })
            validation: Dataset({
                features: ['text'],
                num_rows: 1066
            })
            test: Dataset({
                features: ['text'],
                num_rows: 1066
            })
        })
        ```
        """

        def class_encode_column(self, column: str, include_nulls: bool = False) -> "DatasetDict":
        """Casts the given column as [`~datasets.features.ClassLabel`] and updates the tables.

        Args:
            column (`str`):
                The name of the column to cast.
            include_nulls (`bool`, defaults to `False`):
                Whether to include null values in the class labels. If `True`, the null values will be encoded as the `"None"` class label.

                <Added version="1.14.2"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("boolq")
        >>> ds["train"].features
        {'answer': Value(dtype='bool', id=None),
         'passage': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None)}
        >>> ds = ds.class_encode_column("answer")
        >>> ds["train"].features
        {'answer': ClassLabel(num_classes=2, names=['False', 'True'], id=None),
         'passage': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None)}
        ```
        """

    @contextlib.contextmanager
    def formatted_as(
        self,
        type: Optional[str] = None,
        columns: Optional[List] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """To be used in a `with` statement. Set `__getitem__` return format (type and columns).
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            type (`str`, *optional*):
                Output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.
        """

        def set_format(
        self,
        type: Optional[str] = None,
        columns: Optional[List] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """Set `__getitem__` return format (type and columns).
        The format is set for every dataset in the dataset dictionary.

        Args:
            type (`str`, *optional*):
                Output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects),
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        It is possible to call `map` after calling `set_format`. Since `map` may add new columns, then the list of formatted columns
        gets updated. In this case, if you apply `map` on a dataset to add a new column, then this column will be formatted:

            `new formatted columns = (all columns - previously unformatted columns)`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)
        >>> ds.set_format(type="numpy", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds["train"].format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'numpy'}
        ```
        """

        def reset_format(self):
        """Reset `__getitem__` return format to python objects and all columns.
        The transformation is applied to all the datasets of the dataset dictionary.

        Same as `self.set_format()`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)
        >>> ds.set_format(type="numpy", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds["train"].format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'numpy'}
        >>> ds.reset_format()
        >>> ds["train"].format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': None}
        ```
        """

        def set_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[List] = None,
        output_all_columns: bool = False,
    ):
        """Set ``__getitem__`` return format using this transform. The transform is applied on-the-fly on batches when ``__getitem__`` is called.
        The transform is set for every dataset in the dataset dictionary
        As :func:`datasets.Dataset.set_format`, this can be reset using :func:`datasets.Dataset.reset_format`

        Args:
            transform (`Callable`, optional): user-defined formatting transform, replaces the format defined by :func:`datasets.Dataset.set_format`
                A formatting function is a callable that takes a batch (as a dict) as input and returns a batch.
                This function is applied right before returning the objects in ``__getitem__``.
            columns (`List[str]`, optional): columns to format in the output
                If specified, then the input batch of the transform only contains those columns.
            output_all_columns (`bool`, default to False): keep un-formatted columns as well in the output (as python objects)
                If set to True, then the other un-formatted columns are kept with the output of the transform.

        """

        def with_format(
        self,
        type: Optional[str] = None,
        columns: Optional[List] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ) -> "DatasetDict":
        """Set `__getitem__` return format (type and columns). The data formatting is applied on-the-fly.
        The format `type` (for example "numpy") is used to format batches when using `__getitem__`.
        The format is set for every dataset in the dataset dictionary.

        It's also possible to use custom transforms for formatting using [`~datasets.Dataset.with_transform`].

        Contrary to [`~datasets.DatasetDict.set_format`], `with_format` returns a new [`DatasetDict`] object with new [`Dataset`] objects.

        Args:
            type (`str`, *optional*):
                Output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
        >>> ds["train"].format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': None}
        >>> ds = ds.with_format(type='tensorflow', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds["train"].format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'tensorflow'}
        ```
        """

        def with_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[List] = None,
        output_all_columns: bool = False,
    ) -> "DatasetDict":
        """Set `__getitem__` return format using this transform. The transform is applied on-the-fly on batches when `__getitem__` is called.
        The transform is set for every dataset in the dataset dictionary

        As [`~datasets.Dataset.set_format`], this can be reset using [`~datasets.Dataset.reset_format`].

        Contrary to [`~datasets.DatasetDict.set_transform`], `with_transform` returns a new [`DatasetDict`] object with new [`Dataset`] objects.

        Args:
            transform (`Callable`, *optional*):
                User-defined formatting transform, replaces the format defined by [`~datasets.Dataset.set_format`].
                A formatting function is a callable that takes a batch (as a dict) as input and returns a batch.
                This function is applied right before returning the objects in `__getitem__`.
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                If specified, then the input batch of the transform only contains those columns.
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects).
                If set to `True`, then the other un-formatted columns are kept with the output of the transform.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> def encode(example):
        ...     return tokenizer(example['text'], truncation=True, padding=True, return_tensors="pt")
        >>> ds = ds.with_transform(encode)
        >>> ds["train"][0]
        {'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]),
         'input_ids': tensor([  101,  1103,  2067,  1110, 17348,  1106,  1129,  1103,  6880,  1432,
                112,   188,  1207,   107, 14255,  1389,   107,  1105,  1115,  1119,
                112,   188,  1280,  1106,  1294,   170, 24194,  1256,  3407,  1190,
                170, 11791,  5253,   188,  1732,  7200, 10947, 12606,  2895,   117,
                179,  7766,   118,   172, 15554,  1181,  3498,  6961,  3263,  1137,
                188,  1566,  7912, 14516,  6997,   119,   102]),
         'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0])}
        ```
        """

        def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_names: Optional[Dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> "DatasetDict":
        """Apply a function to all the elements in the table (individually or in batches)
        and update the table (if function does updated examples).
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            function (`callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False`
                - `function(example: Dict[str, Any], indices: int) -> Dict[str, Any]` if `batched=False` and `with_indices=True`
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False`
                - `function(batch: Dict[str, List], indices: List[int]) -> Dict[str, List]` if `batched=True` and `with_indices=True`

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.

            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`[Union[str, List[str]]]`, *optional*, defaults to `None`):
                The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`,
                `batch_size <= 0` or `batch_size == None` then provide the full dataset as a single batch to `function`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`[Union[str, List[str]]]`, *optional*, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_names (`[Dict[str, str]]`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, default `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`[datasets.Features]`, *optional*, defaults to `None`):
                Use a specific [`Features`] to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Disallow null values in the table.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes for multiprocessing. By default it doesn't
                use multiprocessing.
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while mapping examples.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> ds["train"][0:3]["text"]
        ['Review: the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
         'Review: the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
         'Review: effective but too-tepid biopic']

        # process a batch of examples
        >>> ds = ds.map(lambda example: tokenizer(example["text"]), batched=True)
        # set number of processors
        >>> ds = ds.map(add_prefix, num_proc=4)
        ```
        """

        def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_names: Optional[Dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> "DatasetDict":
        """Apply a filter function to all the elements in the table in batches
        and update the table so that the dataset only includes examples according to the filter function.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            function (`Callable`): Callable with one of the following signatures:

                - `function(example: Dict[str, Any]) -> bool` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> bool` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> List[bool]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> List[bool]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                If no function is provided, defaults to an always `True` function: `lambda x: True`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the
                signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`[Union[str, List[str]]]`, *optional*, defaults to `None`):
                The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`
                `batch_size <= 0` or `batch_size == None` then provide the full dataset as a single batch to `function`.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if chaching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_names (`[Dict[str, str]]`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes for multiprocessing. By default it doesn't
                use multiprocessing.
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while filtering examples.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds.filter(lambda x: x["label"] == 1)
        DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 4265
            })
            validation: Dataset({
                features: ['text', 'label'],
                num_rows: 533
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 533
            })
        })
        ```
        """

        def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_names: Optional[Dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        """Create and cache a new Dataset by flattening the indices mapping.

        Args:
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            cache_file_names (`Dict[str, str]`, *optional*, default `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`Optional[datasets.Features]`, defaults to `None`):
                Use a specific [`Features`] to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Allow null values in the table.
            num_proc (`int`, optional, default `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
        """

        def sort(
        self,
        column_names: Union[str, Sequence[str]],
        reverse: Union[bool, Sequence[bool]] = False,
        kind="deprecated",
        null_placement: str = "at_end",
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDict":
        """Create a new dataset sorted according to a single or multiple columns.

        Args:
            column_names (`Union[str, Sequence[str]]`):
                Column name(s) to sort by.
            reverse (`Union[bool, Sequence[bool]]`, defaults to `False`):
                If `True`, sort by descending order rather than ascending. If a single bool is provided,
                the value is applied to the sorting of all column names. Otherwise a list of bools with the
                same length and order as column_names must be provided.
            kind (`str`, *optional*):
                Pandas algorithm for sorting selected in `{quicksort, mergesort, heapsort, stable}`,
                The default is `quicksort`. Note that both `stable` and `mergesort` use timsort under the covers and, in general,
                the actual implementation will vary with data type. The `mergesort` option is retained for backwards compatibility.
                <Deprecated version="2.8.0">

                `kind` was deprecated in version 2.10.0 and will be removed in 3.0.0.

                </Deprecated>
            null_placement (`str`, defaults to `at_end`):
                Put `None` values at the beginning if `at_start` or `first` or at the end if `at_end` or `last`
            keep_in_memory (`bool`, defaults to `False`):
                Keep the sorted indices in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the sorted indices
                can be identified, use it instead of recomputing.
            indices_cache_file_names (`[Dict[str, str]]`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                indices mapping instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                Higher value gives smaller cache files, lower value consume less temporary memory.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset('rotten_tomatoes')
        >>> ds['train']['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> sorted_ds = ds.sort('label')
        >>> sorted_ds['train']['label'][:10]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> another_sorted_ds = ds.sort(['label', 'text'], reverse=[True, False])
        >>> another_sorted_ds['train']['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ```
        """

        def shuffle(
        self,
        seeds: Optional[Union[int, Dict[str, Optional[int]]]] = None,
        seed: Optional[int] = None,
        generators: Optional[Dict[str, np.random.Generator]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDict":
        """Create a new Dataset where the rows are shuffled.

        The transformation is applied to all the datasets of the dataset dictionary.

        Currently shuffling uses numpy random generators.
        You can either supply a NumPy BitGenerator to use, or a seed to initiate NumPy's default random generator (PCG64).

        Args:
            seeds (`Dict[str, int]` or `int`, *optional*):
                A seed to initialize the default BitGenerator if `generator=None`.
                If `None`, then fresh, unpredictable entropy will be pulled from the OS.
                If an `int` or `array_like[ints]` is passed, then it will be passed to SeedSequence to derive the initial BitGenerator state.
                You can provide one `seed` per dataset in the dataset dictionary.
            seed (`int`, *optional*):
                A seed to initialize the default BitGenerator if `generator=None`. Alias for seeds (a `ValueError` is raised if both are provided).
            generators (`Dict[str, *optional*, np.random.Generator]`):
                Numpy random Generator to use to compute the permutation of the dataset rows.
                If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
                You have to provide one `generator` per dataset in the dataset dictionary.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            indices_cache_file_names (`Dict[str, str]`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                indices mappings instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds["train"]["label"][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # set a seed
        >>> shuffled_ds = ds.shuffle(seed=42)
        >>> shuffled_ds["train"]["label"][:10]
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        ```
        """

        def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        fs="deprecated",
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[Dict[str, int]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        """
        Saves a dataset dict to a filesystem using `fsspec.spec.AbstractFileSystem`.

        For [`Image`] and [`Audio`] data:

        All the Image() and Audio() data are stored in the arrow files.
        If you want to store paths or urls, please use the Value("string") type.

        Args:
            dataset_dict_path (`str`):
                Path (e.g. `dataset/train`) or remote URI
                (e.g. `s3://my-bucket/dataset/train`) of the dataset dict directory where the dataset dict will be
                saved to.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem where the dataset will be saved to.

                <Deprecated version="2.8.0">

                `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                </Deprecated>

            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
                (like `"50MB"`).
            num_shards (`Dict[str, int]`, *optional*):
                Number of shards to write. By default the number of shards depends on `max_shard_size` and `num_proc`.
                You need to provide the number of shards for each dataset in the dataset dictionary.
                Use a dictionary to define a different num_shards for each split.

                <Added version="2.8.0"/>
            num_proc (`int`, *optional*, default `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Example:

        ```python
        >>> dataset_dict.save_to_disk("path/to/dataset/directory")
        >>> dataset_dict.save_to_disk("path/to/dataset/directory", max_shard_size="1GB")
        >>> dataset_dict.save_to_disk("path/to/dataset/directory", num_shards={"train": 1024, "test": 8})
        ```
        """

    @staticmethod
    def load_from_disk(
        dataset_dict_path: PathLike,
        fs="deprecated",
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "DatasetDict":
        """
        Load a dataset that was previously saved using [`save_to_disk`] from a filesystem using `fsspec.spec.AbstractFileSystem`.

        Args:
            dataset_dict_path (`str`):
                Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3//my-bucket/dataset/train"`)
                of the dataset dict directory where the dataset dict will be loaded from.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem where the dataset will be saved to.

                <Deprecated version="2.8.0">

                `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                </Deprecated>

            keep_in_memory (`bool`, defaults to `None`):
                Whether to copy the dataset in-memory. If `None`, the
                dataset will not be copied in-memory unless explicitly enabled by setting
                `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
                [improve performance](../cache#improve-performance) section.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> ds = load_from_disk('path/to/dataset/directory')
        ```
        """

    @staticmethod
    def from_csv(
        path_or_paths: Dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from CSV file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the CSV file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (str, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`pandas.read_csv`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_csv({'train': 'path/to/dataset.csv'})
        ```
        """

    @staticmethod
    def from_json(
        path_or_paths: Dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from JSON Lines file(s).

        Args:
            path_or_paths (`path-like` or list of `path-like`):
                Path(s) of the JSON Lines file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (str, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`JsonConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_json({'train': 'path/to/dataset.json'})
        ```
        """

    @staticmethod
    def from_parquet(
        path_or_paths: Dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from Parquet file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the CSV file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            columns (`List[str]`, *optional*):
                If not `None`, only these columns will be read from the file.
                A column name may be a prefix of a nested field, e.g. 'a' will select
                'a.b', 'a.c', and 'a.d.e'.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`ParquetConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_parquet({'train': 'path/to/dataset/parquet'})
        ```
        """

     @staticmethod
    def from_text(
        path_or_paths: Dict[str, PathLike],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ) -> "DatasetDict":
        """Create [`DatasetDict`] from text file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the text file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`TextConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_text({'train': 'path/to/dataset.txt'})
        ```
        """

        def push_to_hub(
        self,
        repo_id,
        config_name: str = "default",
        set_default: Optional[bool] = None,
        data_dir: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        branch="deprecated",
        create_pr: Optional[bool] = False,
        max_shard_size: Optional[Union[int, str]] = None,
        num_shards: Optional[Dict[str, int]] = None,
        embed_external_files: bool = True,
    ) -> CommitInfo:
        """Pushes the [`DatasetDict`] to the hub as a Parquet dataset.
        The [`DatasetDict`] is pushed using HTTP requests and does not need to have neither git or git-lfs installed.

        Each dataset split will be pushed independently. The pushed dataset will keep the original split names.

        The resulting Parquet files are self-contained by default: if your dataset contains [`Image`] or [`Audio`]
        data, the Parquet files will store the bytes of your images or audio files.
        You can disable this by setting `embed_external_files` to False.

        Args:
            repo_id (`str`):
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.
            config_name (`str`):
                Configuration name of a dataset. Defaults to "default".
            set_default (`bool`, *optional*):
                Whether to set this configuration as the default one. Otherwise, the default configuration is the one
                named "default".
            data_dir (`str`, *optional*):
                Directory name that will contain the uploaded data files. Defaults to the `config_name` if different
                from "default", else "data".

                <Added version="2.17.0"/>
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload dataset"`.
            commit_description (`str`, *optional*):
                Description of the commit that will be created.
                Additionally, description of the PR if a PR is created (`create_pr` is True).

                <Added version="2.16.0"/>
            private (`bool`, *optional*):
                Whether the dataset repository should be set to private or not. Only affects repository creation:
                a repository that already exists will not be affected by that parameter.
            token (`str`, *optional*):
                An optional authentication token for the Hugging Face Hub. If no token is passed, will default
                to the token saved locally when logging in with `huggingface-cli login`. Will raise an error
                if no token is passed and the user is not logged-in.
            revision (`str`, *optional*):
                Branch to push the uploaded files to. Defaults to the `"main"` branch.

                <Added version="2.15.0"/>
            branch (`str`, *optional*):
                The git branch on which to push the dataset. This defaults to the default branch as specified
                in your repository, which defaults to `"main"`.

                <Deprecated version="2.15.0">

                `branch` was deprecated in favor of `revision` in version 2.15.0 and will be removed in 3.0.0.

                </Deprecated>
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.

                <Added version="2.15.0"/>
            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
                (like `"500MB"` or `"1GB"`).
            num_shards (`Dict[str, int]`, *optional*):
                Number of shards to write. By default, the number of shards depends on `max_shard_size`.
                Use a dictionary to define a different num_shards for each split.

                <Added version="2.8.0"/>
            embed_external_files (`bool`, defaults to `True`):
                Whether to embed file bytes in the shards.
                In particular, this will do the following before the push for the fields of type:

                - [`Audio`] and [`Image`] removes local path information and embed file content in the Parquet files.

        Return:
            huggingface_hub.CommitInfo

        Example:

        ```python
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", private=True)
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", max_shard_size="1GB")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", num_shards={"train": 1024, "test": 8})
        ```

        If you want to add a new configuration (or subset) to a dataset (e.g. if the dataset has multiple tasks/versions/languages):

        ```python
        >>> english_dataset.push_to_hub("<organization>/<dataset_id>", "en")
        >>> french_dataset.push_to_hub("<organization>/<dataset_id>", "fr")
        >>> # later
        >>> english_dataset = load_dataset("<organization>/<dataset_id>", "en")
        >>> french_dataset = load_dataset("<organization>/<dataset_id>", "fr")
        ```
        """

```

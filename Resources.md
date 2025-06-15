# Resources on Transformer Data Preparation, Trainng, and Fine-tuning 

## Articles on Transformer Model Fine Tuning: repos, online materials, videos

HuggingFace notebook examples: 

  https://github.com/huggingface/notebooks/blob/main/examples

[Universal Language Model Fine-tuning for Text Classification, Jeremy Howard, Sebastian Rudder, FastAI, 2018](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/Universal_Language_Model_Fine-tuning_for_Text_Classification_Howard_2018.pdf)

[Semi-supervised Sequence Learning, A Dai, Quoc V. Le, Google, 2015](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/Semi-supervised_Sequence_Learning_Dai_2015.pdf)

[NLP with Deep Learning, Rohan Paul, 2022, series of 40 youtube videos](https://www.youtube.com/playlist?list=PLxqBkZuBynVTn2lkHNAcw6lgm1MD5QiMK)

## Dense Text Representations

[Introduction to Dense Text Representations - Part 1, Nils Reimers, Jun 21, 2021](https://youtu.be/qmN1fJ7Fdmo?si=sR50ZrGXURzY_weS)

[Introduction to Dense Text Representations - Part 2, Nils Reimers, Jun 21, 2021](https://youtu.be/0RV-q0--NLs?si=8cktLBFigHlNZzi-)

[Introduction to Dense Text Representation - Part 3, Nils Reimers, Jun 21, 2021](https://youtu.be/t4Gf4LruVZ4?si=C2fjB45Vsye0t97p)


## Tokenization

[Tokenization: A Complete Guide, Byte-Pair Encoding, WordPiece, and more by Bradney Smith, Medium, 2024](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/tokenization/Tokenization_A_Complete_Guide_Byte-Pair_Encoding_WordPiece_and_more_by_Bradney_Smith_Medium_2024.pdf)

HuggingFace notebook example: 

  https://github.com/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb 


### Byte-Pair Encoding

[Let's build the GPT Tokenizer, Andrej Karpathy, 2024](https://youtu.be/zduSFxRajkE?si=AOUNH7lcQiZH5FeV)

  Supplementary links:

  Google colab for the video: https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing

  GitHub repo for the video: minBPE: https://github.com/karpathy/minbpe

  tiktokenizer: https://tiktokenizer.vercel.app
  
  tiktoken from OpenAI: https://github.com/openai/tiktoken
  
  sentencepiece from Google: https://github.com/google/sentencepiece

[Byte Pair Encoding Tokenization, HuggingFace, youtube video, 2022](https://www.youtube.com/watch?v=HEikzVL-lZU)

[Byte Pair Encoding, From Languages to Information, youtube video, 2022](https://www.youtube.com/watch?v=tOMjTCO0htA)

## Embeddings

[Sentence Embeddings. Introduction to Sentence Embeddings, https://osanseviero.github.io blog, 2024](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings/)

[Word Embeddings with word2vec from Scratch in Python, Bradney Smith, 2024](https://medium.com/p/eb9326c6ab7c)

[How to train a model to generate image embeddings from scratch, Underfitted, 2024](https://youtu.be/GikIJpUv6oo?si=qlFn69mI-jzw-EDf)

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Nils Reimers and Iryna Gurevych, 2019](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/bert/Sentence-BERT-Sentence_Embeddings_using_Siamese_BERT-Networks_Reimers_2019.pdf)

## Loss Functions

Sentence Transformers Loss Functions: https://sbert.net/docs/package_reference/sentence_transformer/losses.html

[Efficient Natural Language Response Suggestion for Smart Reply, M. Henderson et al, 2017](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/loss_functions/Efficient_Natural_Language_Response_Suggestion_for_Smart_Reply_Henderson_2017.pdf)

## Fine-tuning a pretrained model

HuggingFace notebook example on training: 

  https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb

HuggingFace notebook example on fine-tunning for classification: 

  https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb

HuggingFace notebook example on fine-tuning for question-answering:

  https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb


## Attention

[Self-Attention Explained with Code, Bradney Smith, Medium, 2024](https://medium.com/data-science/contextual-transformer-embeddings-using-self-attention-explained-with-diagrams-and-python-code-d7a9f0f4d94e)

## BERT and SBERT 

[BERT Explained](https://youtu.be/xI0HHN5XKDo?si=CJLDvv8Fd13A9Ri6)

[SBERT Explained, CodeEmporium, 2022](https://youtu.be/O3xbVmpdJwU?si=v9X7xjFylkEi-HSB)

SBERT docs: https://sbert.net/

[A Complete Guide to BERT with Code by Bradney Smith, Medium, 2024](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/bert/A_Complete_Guide_to_BERT_with_Code_by_Bradney_Smith_Medium_2024.pdf)

### SBERT/BERT embeddings

[SBERT (Sentence Transformers) is not BERT Sentence Embedding: Intro & Tutorial, Discover AI, 2023](https://youtu.be/lVqwznaVi78?si=-MJoRX51Z7P9un4P)

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Nils Reimers and Iryna Gurevych, 2019](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/bert/Sentence-BERT-Sentence_Embeddings_using_Siamese_BERT-Networks_Reimers_2019.pdf)

### SBERT/BERT Loss functions 

[Training State-of-the-Art Sentence Embedding Models, Nils Reimers, youtube video, Jun 30, 2021](https://youtu.be/RHXZKUr8qOY?si=5PIDz7nPaWtCwoy1)

SBERT's MultipleNegativesRankingLoss: https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss

### BERT/SBERT fine-tuning


[Python Tutorial to Fine-tune SBERT BI-Encoder with Domain-specific Training Dataset, Discover AI, youtube video, 2023](https://youtu.be/FidMAm-tj9k?si=oED-7avcJFsMrLyv)

[Fine-Tune SBERT on specific Knowledge Domain with Cross-Encoder Sentence Transformers, Discover AI, youtube video, 2023](https://youtu.be/JxfS5ZjdxGE?si=a87k5dtQzQu1qTu8)

[Fine-Tuning BERT for Text Classification (with Example Code), Shaw Talebi, 2024](https://youtu.be/4QHg8Ix8WWQ?si=DkQyws-ZPtiOJ5zS)

  repo for the Phishing classification example: https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs/model-compression

  model for the Phishing classification example: https://huggingface.co/shawhin/bert-phishing-classifier_teacher

  dataset for the Phishing classification example: https://huggingface.co/datasets/shawhin/phishing-site-classification

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, J. Devlin et al, 2019](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/bert/BERT-Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding_Devlin_2019.pdf)

### Mistral fine-tuning

[Mistral 7B Explained: Towards More Efficient Language Models, Bradney Smith, Medium, 2024](https://github.com/dimitarpg13/transformer_finetuning/blob/main/articles/mistral/Mistral_7B_Explained_Towards_More_Efficient_Language_Models_by_Bradney_Smith_Medium_2024.pdf)

### BERT pre-training

[Pre-Train BERT from scratch: Solution for Company Domain Knowledge Data, Discover AI, youtube video, 2023](https://youtu.be/IcrN_L2w0_Y?si=C4mWIUrcxk-8HByx)

### Semantic Search with SBERT

[Semantic Search with Sentence Transformers, sbert.net README](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)

[Semantic Search with Sentence Transformers, Discover AI, 2022](https://youtu.be/ewlCCB7EFPs?si=39x3WjNZQIyofWUm)



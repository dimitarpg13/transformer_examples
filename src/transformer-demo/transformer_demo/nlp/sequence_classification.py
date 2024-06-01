from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


def main():
    imdb = load_dataset("imdb")
    print(imdb["test"][0])

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # TODO: finish the script


if __name__ == "__main__":
    main()

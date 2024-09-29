import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Chapter 2: Working with Text""")
    return


@app.cell
def __(mo):
    from importlib.metadata import version

    mo.vstack(
        [
            "torch version: {}".format(version("torch")),
            "tiktoken version: {}".format(version("tiktoken")),
        ]
    )
    return (version,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.2 Tokenizing text""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Use [The Verdict by Edith Wharton](https://en.wikisource.org/wiki/The_Verdict), a public domain short story""")
    return


@app.cell
def __(__file__):
    import os
    import urllib.request

    story_file_path = os.path.join(os.path.dirname(__file__), "the-verdict.txt")

    if not os.path.exists(story_file_path):
        _url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        urllib.request.urlretrieve(_url, story_file_path)
    return os, story_file_path, urllib


@app.cell
def __(mo, story_file_path):
    with open(story_file_path, "r", encoding="utf-8") as _f:
        story_raw_text = _f.read()

    mo.vstack(
        [
            "Total number of character: {}".format(len(story_raw_text)),
            story_raw_text[:99],
        ]
    )
    return (story_raw_text,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Split on whitespace:""")
    return


@app.cell
def __():
    import re

    sample1_text = "Hello, world. This, is a test."
    _result = re.split(r"(\s)", sample1_text)

    _result
    return re, sample1_text


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Split on whitespace and commas and periods:""")
    return


@app.cell
def __(re, sample1_text):
    sample1_result = re.split(r"([,.]|\s)", sample1_text)

    sample1_result
    return (sample1_result,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Remove empty strings:""")
    return


@app.cell
def __(sample1_result):
    # Strip whitespace from each item and then filter out any empty strings.
    _striped_result = [item for item in sample1_result if item.strip()]

    _striped_result
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Handle other types of punctuation:""")
    return


@app.cell
def __(re):
    _text = "Hello, world. Is this-- a test?"

    _result = re.split(r'([,.:;?_!"()\']|--|\s)', _text)
    _result = [item.strip() for item in _result if item.strip()]

    _result
    return


@app.cell
def __(re, story_raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', story_raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    preprocessed[:30]
    return (preprocessed,)


@app.cell
def __(preprocessed):
    "number of tokens: {}".format(len(preprocessed))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.3 Converting tokens into token IDs""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""From these tokens, we can now build a vocabulary that consists of all the unique tokens""")
    return


@app.cell
def __(preprocessed):
    all_words = sorted(set(preprocessed))
    _vocab_size = len(all_words)

    _vocab_size
    return (all_words,)


@app.cell
def __(all_words):
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return (vocab,)


@app.cell
def __(vocab):
    for _i, _item in enumerate(vocab.items()):
        print(_item)
        if _i >= 50:
            break
    return


@app.cell
def __(re):
    class SimpleTokenizerV1:
        def __init__(self, vocab: dict[str, int]):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text: str):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids: list[int]):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
            return text
    return (SimpleTokenizerV1,)


@app.cell
def __(SimpleTokenizerV1, vocab):
    tokenizer1 = SimpleTokenizerV1(vocab)

    sample2_text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    sample2_ids = tokenizer1.encode(sample2_text)
    sample2_ids
    return sample2_ids, sample2_text, tokenizer1


@app.cell
def __(sample2_ids, tokenizer1):
    tokenizer1.decode(sample2_ids)
    return


@app.cell
def __(sample2_text, tokenizer1):
    tokenizer1.decode(tokenizer1.encode(sample2_text))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.4 Adding special context tokens""")
    return


@app.cell
def __(SimpleTokenizerV1, vocab):
    _tokenizer = SimpleTokenizerV1(vocab)

    _text = "Hello, do you like tea. Is this-- a test?"

    # Expect an error since "Hello" is not in dictionary.
    # _tokenizer.encode(_text)
    return


@app.cell
def __(preprocessed):
    _all_tokens = sorted(list(set(preprocessed)))
    _all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    extended_vocab = {token: integer for integer, token in enumerate(_all_tokens)}
    return (extended_vocab,)


@app.cell
def __(extended_vocab):
    len(extended_vocab.items())
    return


@app.cell
def __(extended_vocab, mo):
    with mo.redirect_stdout():
        for _i, _item in enumerate(list(extended_vocab.items())[-5:]):
            print(_item)
    return


@app.cell
def __(re):
    class SimpleTokenizerV2:
        def __init__(self, vocab: dict[str, int]):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text: str):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [
                item if item in self.str_to_int else "<|unk|>"
                for item in preprocessed
            ]

            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids: list[int]):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
            return text
    return (SimpleTokenizerV2,)


@app.cell
def __(SimpleTokenizerV2, extended_vocab):
    tokenizer2 = SimpleTokenizerV2(extended_vocab)

    _text1 = "Hello, do you like tea?"
    _text2 = "In the sunlit terraces of the palace."

    sample3_text = " <|endoftext|> ".join((_text1, _text2))

    sample3_text
    return sample3_text, tokenizer2


@app.cell
def __(sample3_text, tokenizer2):
    tokenizer2.encode(sample3_text)
    return


@app.cell
def __(sample3_text, tokenizer2):
    tokenizer2.decode(tokenizer2.encode(sample3_text))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.5 BytePair encoding""")
    return


@app.cell
def __():
    import tiktoken
    return (tiktoken,)


@app.cell
def __(tiktoken):
    tokenizer_gpt2 = tiktoken.get_encoding("gpt2")
    return (tokenizer_gpt2,)


@app.cell
def __(tokenizer_gpt2):
    _text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )

    integers_gpt = tokenizer_gpt2.encode(_text, allowed_special={"<|endoftext|>"})

    integers_gpt
    return (integers_gpt,)


@app.cell
def __(integers_gpt, tokenizer_gpt2):
    _strings = tokenizer_gpt2.decode(integers_gpt)

    _strings
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""#### Exercise 2.1""")
    return


@app.cell
def __(tokenizer_gpt2):
    example_text = "Akwirw ier"
    example_encoding = tokenizer_gpt2.encode(example_text)
    example_encoding
    return example_encoding, example_text


@app.cell
def __(example_encoding, tokenizer_gpt2):
    [tokenizer_gpt2.decode([x]) for x in example_encoding]
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.6 Data sampling with a sliding window""")
    return


@app.cell
def __(story_raw_text, tokenizer_gpt2):
    story_enc_text = tokenizer_gpt2.encode(story_raw_text)
    len(story_enc_text)
    return (story_enc_text,)


@app.cell
def __(story_enc_text):
    story_enc_sample = story_enc_text[50:]
    return (story_enc_sample,)


@app.cell
def __(mo, story_enc_sample):
    context_size = 4

    _x = story_enc_sample[:context_size]
    _y = story_enc_sample[1 : context_size + 1]

    with mo.redirect_stdout():
        print(f"x: {_x}")
        print(f"y:      {_y}")
    return (context_size,)


@app.cell
def __(context_size, mo, story_enc_sample):
    with mo.redirect_stdout():
        for _i in range(1, context_size + 1):
            _context = story_enc_sample[:_i]
            _desired = story_enc_sample[_i]

            print(f"{_context} ----> {_desired}")
    return


@app.cell
def __(context_size, mo, story_enc_sample, tokenizer_gpt2):
    with mo.redirect_stdout():
        for _i in range(1, context_size + 1):
            _context = story_enc_sample[:_i]
            _desired = story_enc_sample[_i]

            print(
                f"{tokenizer_gpt2.decode(_context)} ----> {tokenizer_gpt2.decode([_desired])}"
            )
    return


@app.cell
def __():
    import torch

    "PyTorch version: {}".format(torch.__version__)
    return (torch,)


@app.cell
def __(tiktoken, torch):
    from torch.utils.data import Dataset, DataLoader


    class GPTDatasetV1(Dataset):
        def __init__(
            self,
            txt: str,
            tokenizer: tiktoken.Encoding,
            max_length: int,
            stride: int,
        ):
            self.input_ids = []
            self.target_ids = []

            # Tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i : i + max_length]
                target_chunk = token_ids[i + 1 : i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx: int):
            return self.input_ids[idx], self.target_ids[idx]
    return DataLoader, Dataset, GPTDatasetV1


@app.cell
def __(DataLoader, GPTDatasetV1, tiktoken):
    def create_dataloader_v1(
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader
    return (create_dataloader_v1,)


@app.cell
def __(create_dataloader_v1, story_raw_text):
    _dataloader = create_dataloader_v1(
        story_raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    _data_iter = iter(_dataloader)
    _first_batch = next(_data_iter)
    print(_first_batch)
    _second_batch = next(_data_iter)
    print(_second_batch)
    return


@app.cell
def __(create_dataloader_v1, story_raw_text):
    _dataloader = create_dataloader_v1(
        story_raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )

    _data_iter = iter(_dataloader)
    _inputs, _targets = next(_data_iter)
    print("Inputs:\n", _inputs)
    print("\nTargets:\n", _targets)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.7 Creating token embeddings""")
    return


@app.cell
def __(torch):
    input_ids = torch.tensor([2, 3, 5, 1])
    return (input_ids,)


@app.cell
def __(torch):
    _vocab_size = 6
    _output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    return (embedding_layer,)


@app.cell
def __(embedding_layer):
    embedding_layer.weight
    return


@app.cell
def __(embedding_layer, torch):
    embedding_layer(torch.tensor([3]))
    return


@app.cell
def __(embedding_layer, input_ids):
    embedding_layer(input_ids)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 2.8 Encoding word positions""")
    return


@app.cell
def __(torch):
    _vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(_vocab_size, output_dim)
    return output_dim, token_embedding_layer


@app.cell
def __(create_dataloader_v1, story_raw_text):
    max_length = 4
    _dataloader = create_dataloader_v1(
        story_raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    _data_iter = iter(_dataloader)
    inputs, _targets = next(_data_iter)
    return inputs, max_length


@app.cell
def __(inputs, mo):
    with mo.redirect_stdout():
        print("Token IDs:\n{}".format(inputs))
        print("\nInputs shape:\n{}".format(inputs.shape))
    return


@app.cell
def __(inputs, mo, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs)
    mo.plain_text(token_embeddings.shape)
    return (token_embeddings,)


@app.cell
def __(max_length, output_dim, torch):
    _context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(_context_length, output_dim)
    return (pos_embedding_layer,)


@app.cell
def __(max_length, mo, pos_embedding_layer, torch):
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    mo.plain_text(pos_embeddings.shape)
    return (pos_embeddings,)


@app.cell
def __(mo, pos_embeddings, token_embeddings):
    _input_embeddings = token_embeddings + pos_embeddings
    mo.plain_text(_input_embeddings.shape)
    return


if __name__ == "__main__":
    app.run()

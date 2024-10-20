import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Chapter 2: Bonus - Data sampling with a sliding window with number data""")
    return


@app.cell
def __():
    from importlib.metadata import version
    import torch

    "torch version: {}".format(version("torch"))
    return torch, version


@app.cell
def __():
    with open("number-data.txt", "w", encoding="utf-8") as _f:
        for number in range(1001):
            _f.write(f"{number} ")
    return (number,)


@app.cell
def __():
    from torch.utils.data import DataLoader
    from llm_from_scratch.ch_02_d_dataloader_intuition.gpt_dataset_v1 import (
        GPTDatasetV1,
    )


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
        # tokenizer = tiktoken.get_encoding("gpt2")
        tokenizer = None

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
    return DataLoader, GPTDatasetV1, create_dataloader_v1


@app.cell
def __():
    with open("number-data.txt", "r", encoding="utf-8") as _f:
        raw_text = _f.read()
    return (raw_text,)


@app.cell
def __(create_dataloader_v1, raw_text):
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

    data_iter = iter(dataloader)
    _first_batch = next(data_iter)
    _first_batch
    return data_iter, dataloader


@app.cell
def __(data_iter):
    _second_batch = next(data_iter)
    _second_batch
    return


@app.cell
def __(data_iter):
    _third_batch = next(data_iter)
    _third_batch
    return


@app.cell
def __(dataloader):
    for _batch in dataloader:
        pass

    _last_batch = _batch
    _last_batch
    return


@app.cell
def __(create_dataloader_v1, mo, raw_text, torch):
    torch.manual_seed(123)
    _dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)

    for _inputs, _targets in _dataloader:
        pass

    mo.output.append("Inputs: {}".format(_inputs))
    mo.output.append("Targets: {}".format(_targets))
    return


if __name__ == "__main__":
    app.run()

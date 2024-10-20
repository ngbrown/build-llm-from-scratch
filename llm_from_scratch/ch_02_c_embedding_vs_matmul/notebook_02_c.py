import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Chapter 2: Bonus - Understanding the Difference Between Embedding Layers and Linear Layers""")
    return


@app.cell
def __():
    import torch

    "PyTorch version: {}".format(torch.__version__)
    return (torch,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Using nn.Embedding""")
    return


@app.cell
def __(torch):
    # Suppose we have the following 3 training examples,
    # which may represent token IDs in a LLM context
    _idx = torch.tensor([2, 3, 1])

    # The number of rows in the embedding matrix can be determined
    # by obtaining the largest token ID + 1.
    # If the highest token ID is 3, then we want 4 rows, for the possible
    # token IDs 0, 1, 2, 3
    num_idx = max(_idx) + 1

    # The desired embedding dimension is a hyperparameter
    out_dim = 5
    return num_idx, out_dim


@app.cell
def __(num_idx, out_dim, torch):
    # We use the random seed for reproducibility since
    # weights in the embedding layer are initialized with
    # small random values
    torch.manual_seed(123)

    embedding = torch.nn.Embedding(num_idx, out_dim)
    return (embedding,)


@app.cell
def __(embedding):
    embedding.weight
    return


@app.cell
def __(embedding, torch):
    embedding(torch.tensor([1]))
    return


@app.cell
def __(embedding, torch):
    embedding(torch.tensor([2]))
    return


@app.cell
def __(embedding, torch):
    idx = torch.tensor([2, 3, 1])
    embedding(idx)
    return (idx,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Using nn.Linear""")
    return


@app.cell
def __(idx, torch):
    onehot = torch.nn.functional.one_hot(idx)
    onehot
    return (onehot,)


@app.cell
def __(num_idx, out_dim, torch):
    torch.manual_seed(123)
    linear = torch.nn.Linear(num_idx, out_dim, bias=False)
    linear.weight
    return (linear,)


@app.cell
def __(embedding, linear, torch):
    linear.weight = torch.nn.Parameter(embedding.weight.T)
    return


@app.cell
def __(linear, onehot):
    linear(onehot.float())
    return


@app.cell
def __(embedding, idx):
    embedding(idx)
    return


if __name__ == "__main__":
    app.run()

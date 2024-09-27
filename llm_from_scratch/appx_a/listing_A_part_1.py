import marimo

__generated_with = "0.8.21"
app = marimo.App(width="medium")


@app.cell
def __():
    import torch
    return (torch,)


@app.cell
def __(torch):
    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1, 2, 3])
    tensor2d = torch.tensor([[1, 2],
                             [3, 4]])
    tensor3d = torch.tensor([[[1, 2], [3, 4]],
                             [[5, 6], [7, 8]]])
    return tensor0d, tensor1d, tensor2d, tensor3d


@app.cell
def __(tensor3d):
    tensor3d
    return


if __name__ == "__main__":
    app.run()

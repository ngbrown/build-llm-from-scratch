import marimo

__generated_with = "0.8.21"
app = marimo.App(width="medium")


@app.cell
def __():
    import torch
    print(f'Cuda is available: {torch.cuda.is_available()}')
    return (torch,)


if __name__ == "__main__":
    app.run()

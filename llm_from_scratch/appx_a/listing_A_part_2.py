import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""## Appendix A: Introduction to PyTorch (Part 2)""")
    return


@app.cell
def __(mo):
    mo.md(r"""### A.9 Optimizing training performance with GPUs""")
    return


@app.cell
def __(mo):
    mo.md(r"""### A.9.1 PyTorch computations on GPU devices""")
    return


@app.cell
def __():
    import torch

    torch.__version__
    return (torch,)


@app.cell
def __(torch):
    torch.cuda.is_available()
    return


@app.cell
def __(torch):
    tensor_1 = torch.tensor([1.0, 2.0, 3.0])
    tensor_2 = torch.tensor([4.0, 5.0, 6.0])

    tensor_1 + tensor_2
    return tensor_1, tensor_2


@app.cell
def __(tensor_1, tensor_2):
    gpu_tensor_1 = tensor_1.to("cuda")
    gpu_tensor_2 = tensor_2.to("cuda")

    gpu_tensor_1 + gpu_tensor_2
    return gpu_tensor_1, gpu_tensor_2


@app.cell
def __(gpu_tensor_1):
    _cpu_tensor_1 = gpu_tensor_1.to("cpu")

    # expect error because tensors are not on the same device:
    # _cpu_tensor_1 + gpu_tensor_2
    return


@app.cell
def __(mo):
    mo.md(r"""### A.9.2 Single-GPU training""")
    return


@app.cell
def __(torch):
    X_train = torch.tensor(
        [
            [-1.2, 3.1],
            [-0.9, 2.9],
            [-0.5, 2.6],
            [2.3, -1.1],
            [2.7, -1.5],
        ]
    )

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )

    y_test = torch.tensor([0, 1])
    return X_test, X_train, y_test, y_train


@app.cell
def __(X_test, X_train, y_test, y_train):
    from llm_from_scratch.appx_a.toy_dataset import ToyDataset


    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    return ToyDataset, test_ds, train_ds


@app.cell
def __(test_ds, torch, train_ds):
    from torch.utils.data import DataLoader

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=1,
    )
    return DataLoader, test_loader, train_loader


@app.cell
def __(torch):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()

            self.layers = torch.nn.Sequential(
                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),
                # 2nd hidden layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),
                # output layer
                torch.nn.Linear(20, num_outputs),
            )

        def forward(self, x):
            logits = self.layers(x)
            return logits
    return (NeuralNetwork,)


@app.cell
def __(NeuralNetwork, torch, train_loader):
    import torch.nn.functional as F

    torch.manual_seed(123)
    cpu_model = NeuralNetwork(num_inputs=2, num_outputs=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # NEW
    gpu_model = cpu_model.to(device)  # NEW

    _optimizer = torch.optim.SGD(gpu_model.parameters(), lr=0.5)

    _num_epochs = 3

    for _epoch in range(_num_epochs):
        gpu_model.train()
        for _batch_idx, (_features, _labels) in enumerate(train_loader):
            _features, _labels = _features.to(device), _labels.to(device)  # NEW
            _logits = gpu_model(_features)

            _loss = F.cross_entropy(_logits, _labels)  # Loss function

            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

            ### LOGGING
            print(
                f"Epoch: {_epoch+1:03d}/{_num_epochs:03d}"
                f" | Batch {_batch_idx:03d}/{len(train_loader):03d}"
                f" | Train/Val Loss: {_loss:.2f}"
            )

        gpu_model.eval()
        # Optional model evaluation
    return F, cpu_model, device, gpu_model


@app.cell
def __(DataLoader, torch):
    def compute_accuracy(
        model: torch.nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        model = model.eval()
        correct = 0.0
        total_examples = 0

        for idx, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)  # New

            with torch.no_grad():
                logits = model(features)

            predictions = torch.argmax(logits, dim=1)
            compare = labels == predictions
            correct += torch.sum(compare)
            total_examples += len(compare)

        return (correct / total_examples).item()
    return (compute_accuracy,)


@app.cell
def __(compute_accuracy, device, gpu_model, train_loader):
    compute_accuracy(gpu_model, train_loader, device=device)
    return


@app.cell
def __(compute_accuracy, device, gpu_model, test_loader):
    compute_accuracy(gpu_model, test_loader, device=device)
    return


if __name__ == "__main__":
    app.run()

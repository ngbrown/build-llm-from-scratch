import marimo

__generated_with = "0.8.21"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""## Appendix A: Introduction to PyTorch""")
    return


@app.cell
def __(mo):
    import torch

    mo.plain_text(torch.__version__)
    return (torch,)


@app.cell
def __(torch):
    print(torch.cuda.is_available())
    return


@app.cell
def __(mo):
    mo.md(r"""### A.2 Understanding tensors""")
    return


@app.cell
def __(mo):
    mo.md(r"""### A.2.1 Scalars, vectors, matrices, and tensors""")
    return


@app.cell
def __(torch):
    import numpy as np

    # create a 0D tensor (scalar) from a Python integer
    _tensor0d = torch.tensor(1)

    # create a 1D tensor (vector) from a Python list
    _tensor1d = torch.tensor([1, 2, 3])

    # create a 2D tensor from a nested Python list
    _tensor2d = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )

    # create a 3D tensor from a nested Python list
    _tensor3d_1 = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )

    # create a 3D tensor from NumPy array
    ary3d = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )
    tensor3d_2 = torch.tensor(ary3d)  # Copies NumPy array
    tensor3d_3 = torch.from_numpy(ary3d)  # Shares memory with NumPy array
    return ary3d, np, tensor3d_2, tensor3d_3


@app.cell
def __(ary3d, tensor3d_2):
    ary3d[0, 0, 0] = 999
    tensor3d_2  # remains unchanged
    return


@app.cell
def __(tensor3d_3):
    tensor3d_3  # changes because of memory sharing
    return


@app.cell
def __(mo):
    mo.md(r"""### A.2.2 Tensor data types""")
    return


@app.cell
def __(torch):
    tensor1d = torch.tensor([1, 2, 3])
    tensor1d.dtype
    return (tensor1d,)


@app.cell
def __(torch):
    _floatvec = torch.tensor([1.0, 2.0, 3.0])
    _floatvec.dtype
    return


@app.cell
def __(tensor1d, torch):
    _floatvec = tensor1d.to(torch.float32)
    _floatvec.dtype
    return


@app.cell
def __(mo):
    mo.md(r"""### A.2.3 Common PyTorch tensor operations""")
    return


@app.cell
def __(torch):
    tensor2d = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    tensor2d
    return (tensor2d,)


@app.cell
def __(mo, tensor2d):
    mo.plain_text(tensor2d.shape)
    return


@app.cell
def __(tensor2d):
    tensor2d.reshape(3, 2)
    return


@app.cell
def __(tensor2d):
    tensor2d.view(3, 2)
    return


@app.cell
def __(tensor2d):
    tensor2d.T
    return


@app.cell
def __(tensor2d):
    tensor2d.matmul(tensor2d.T)
    return


@app.cell
def __(tensor2d):
    tensor2d @ tensor2d.T
    return


@app.cell
def __(mo):
    mo.md(r"""### A.3 Seeing models as computation graphs""")
    return


@app.cell
def __():
    import torch.nn.functional as F
    return (F,)


@app.cell
def __(F, torch):
    _y = torch.tensor([1.0])  # true label
    _x1 = torch.tensor([1.1])  # input feature
    _w1 = torch.tensor([2.2])  # weight parameter
    _b = torch.tensor([0.0])  # bias unit

    _z = _x1 * _w1 + _b  # net input
    _a = torch.sigmoid(_z)  # activation & output

    _loss = F.binary_cross_entropy(_a, _y)
    _loss
    return


@app.cell
def __(mo):
    mo.md(r"""### A.4 Automatic differentiation made easy""")
    return


@app.cell
def __():
    from torch.autograd import grad
    return (grad,)


@app.cell
def __(F, grad, mo, torch):
    _y = torch.tensor([1.0])
    _x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    _z = _x1 * w1 + b
    _a = torch.sigmoid(_z)

    loss = F.binary_cross_entropy(_a, _y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)

    mo.plain_text((grad_L_w1, grad_L_b))
    return b, grad_L_b, grad_L_w1, loss, w1


@app.cell
def __(b, loss, mo, w1):
    loss.backward()

    mo.plain_text((w1.grad, b.grad))
    return


@app.cell
def __(mo):
    mo.md(r"""### A.5 Implementing multilayer neural networks""")
    return


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
def __(NeuralNetwork, torch):
    torch.manual_seed(123)
    model_1 = NeuralNetwork(50, 3)
    model_1
    return (model_1,)


@app.cell
def __(model_1):
    _num_params = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
    "Total number of trainable model parameters: {}".format(_num_params)
    return


@app.cell
def __(model_1):
    model_1.layers[0].weight
    return


@app.cell
def __(mo, model_1):
    mo.plain_text(model_1.layers[0].weight.shape)
    return


@app.cell
def __(model_1, torch):
    torch.manual_seed(123)

    X = torch.rand((1, 50))
    _out = model_1(X)
    _out
    return (X,)


@app.cell
def __(X, model_1, torch):
    with torch.no_grad():
        _out = model_1(X)
    _out
    return


@app.cell
def __(X, model_1, torch):
    with torch.no_grad():
        _out = torch.softmax(model_1(X), dim=1)
    _out
    return


@app.cell
def __(mo):
    mo.md(r"""### A.6 Setting up efficient data loaders""")
    return


@app.cell
def __(torch):
    # input
    X_train = torch.tensor(
        [
            [-1.2, 3.1],
            [-0.9, 2.9],
            [-0.5, 2.6],
            [2.3, -1.1],
            [2.7, -1.5],
        ]
    )

    # target
    y_train = torch.tensor([0, 0, 0, 1, 1])
    return X_train, y_train


@app.cell
def __(torch):
    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )

    y_test = torch.tensor([0, 1])
    return X_test, y_test


@app.cell
def __(X_test, X_train, y_test, y_train):
    from torch.utils.data import Dataset


    class ToyDataset(Dataset):
        def __init__(self, X, y):
            self.features = X
            self.labels = y

        def __getitem__(self, index):
            one_x = self.features[index]
            one_y = self.labels[index]
            return one_x, one_y

        def __len__(self):
            return self.labels.shape[0]


    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    return Dataset, ToyDataset, test_ds, train_ds


@app.cell
def __(train_ds):
    len(train_ds)
    return


@app.cell
def __(test_ds, torch, train_ds):
    from torch.utils.data import DataLoader

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    return DataLoader, test_loader, train_loader


@app.cell
def __(mo, train_loader):
    mo.vstack(
        [
            (mo.plain_text(f"Batch {idx+1}:{x}{y}"))
            for idx, (x, y) in enumerate(train_loader)
        ]
    )
    return


@app.cell
def __(DataLoader, train_ds):
    equalsize_train_loader = DataLoader(
        dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )
    return (equalsize_train_loader,)


@app.cell
def __(equalsize_train_loader, mo):
    mo.vstack(
        [
            (mo.plain_text(f"Batch {idx+1}:{x}{y}"))
            for idx, (x, y) in enumerate(equalsize_train_loader)
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(r"""### A.7 A typical training loop""")
    return


@app.cell
def __(F, NeuralNetwork, equalsize_train_loader, torch):
    torch.manual_seed(123)
    model_2 = NeuralNetwork(num_inputs=2, num_outputs=2)
    _optimizer = torch.optim.SGD(model_2.parameters(), lr=0.5)

    _num_epochs = 3

    for _epoch in range(_num_epochs):
        model_2.train()
        for _batch_idx, (_features, _labels) in enumerate(equalsize_train_loader):
            _logits = model_2(_features)

            _loss = F.cross_entropy(_logits, _labels)  # Loss function

            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

            ### LOGGING
            print(
                f"Epoch: {_epoch+1:03d}/{_num_epochs:03d}"
                f" | Batch {_batch_idx:03d}/{len(equalsize_train_loader):03d}"
                f" | Train/Val Loss: {_loss:.2f}"
            )

        model_2.eval()
        # Optional model evaluation
    return (model_2,)


@app.cell
def __(model_2):
    model_2
    return


@app.cell
def __(mo):
    mo.md(r"""Exercise A.3: How many parameters does the neural network have?""")
    return


@app.cell
def __():
    (2 + 1) * 30 + (30 + 1) * 20 + (20 + 1) * 2
    return


@app.cell
def __(model_2):
    _num_params = sum(p.numel() for p in model_2.parameters() if p.requires_grad)
    _num_params
    return


@app.cell
def __(X_train, model_2, torch):
    model_2.eval()

    with torch.no_grad():
        outputs = model_2(X_train)

    outputs
    return (outputs,)


@app.cell
def __(outputs, torch):
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    probas
    return (probas,)


@app.cell
def __(probas, torch):
    predictions = torch.argmax(probas, dim=1)
    predictions
    return (predictions,)


@app.cell
def __(predictions, y_train):
    predictions == y_train
    return


@app.cell
def __(predictions, torch, y_train):
    torch.sum(predictions == y_train)
    return


@app.cell
def __(DataLoader, torch):
    def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader) -> float:
        model = model.eval()
        correct = 0.0
        total_examples = 0

        for idx, (features, labels) in enumerate(dataloader):
            with torch.no_grad():
                logits = model(features)

            predictions = torch.argmax(logits, dim=1)
            compare = labels == predictions
            correct += torch.sum(compare)
            total_examples += len(compare)

        return (correct / total_examples).item()
    return (compute_accuracy,)


@app.cell
def __(compute_accuracy, model_2, train_loader):
    compute_accuracy(model_2, train_loader)
    return


@app.cell
def __(compute_accuracy, model_2, test_loader):
    compute_accuracy(model_2, test_loader)
    return


@app.cell
def __(mo):
    mo.md(r"""### A.8 Saving and loading models""")
    return


@app.cell
def __(model_2, torch):
    torch.save(model_2.state_dict(), "model_a_1.pth")
    return


@app.cell
def __(NeuralNetwork, compute_accuracy, test_loader, torch):
    _model = NeuralNetwork(2, 2) # needs to match the original model exactly
    _model.load_state_dict(torch.load("model_a_1.pth", weights_only=True))
    compute_accuracy(_model, test_loader)
    return


if __name__ == "__main__":
    app.run()

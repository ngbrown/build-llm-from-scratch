# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Union

# NEW imports:
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
def ddp_setup(rank: int, world_size: int):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # rank of machine running rank:0 process
    # here, we assume all GPUs are on the same machine
    os.environ["MASTER_ADDR"] = "localhost"
    # any free port on the machine
    os.environ["MASTER_PORT"] = "12345"
    # Disable libuv because PyTorch for Windows isn't built with support
    os.environ["USE_LIBUV"] = "0"

    # initialize process group
    # Windows users may have to use "gloo" instead of "nccl" as backend
    # nccl: NVIDIA Collective Communication Library
    # gloo: Facebook Collective Communication Library
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


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


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
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


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds),  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# NEW: wrapper
def main(rank: int, world_size: int, num_epochs: int):
    ddp_setup(rank, world_size)  # NEW: initialize process groups

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # the core model is now accessible as model.module

    for epoch in range(num_epochs):

        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    destroy_process_group()  # NEW: cleanly exit distributed mode


def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: Union[torch.device, int]) -> float:
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)

    # NEW: spawn new processes
    # note that spawn will automatically pass the rank
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size spawns one process per GPU

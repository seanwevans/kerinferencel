#!/usr/bin/env python3

""" train.py - trains a model for ebpf inference """

import argparse
import sys

import numpy as np
import torch
import torch.ao.quantization as quant
from torch.ao.quantization.observer import PerChannelMinMaxObserver, default_observer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class TinyPerceptron(nn.Module):
    """A tiny leaky-relu based single-layer perceptron"""

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.quant = quant.QuantStub()
        self.fc1 = nn.Linear(784, 32)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.fc2 = nn.Linear(32, 10)
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x


def parse_args(args):
    """parse command line arguments"""

    argp = argparse.ArgumentParser()

    argp.add_argument("--batch-size", type=int, default=64)
    argp.add_argument("--num-epochs", type=int, default=10)
    argp.add_argument("--learn-rate", type=float, default=1e-3)
    argp.add_argument("--input-size", type=int, default=784)
    argp.add_argument("--hidden-size", type=int, default=32)
    argp.add_argument("--output-size", type=int, default=10)
    argp.add_argument("--leaky-slope", type=float, default=1e-2)

    return argp.parse_args(args)


def train(model, device, train_loader, optimizer, epochs=5):
    """train the model"""

    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


def evaluate(model, device, data_loader):
    """evaluate the model"""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def export_quantized_parameters(model, prefix=""):
    """save quantized model"""

    fc1 = model.fc1
    fc2 = model.fc2

    fc1_weight = fc1.weight().int_repr().detach().cpu().numpy()
    fc1_bias = fc1.bias().detach().cpu().numpy()  # should be int32
    fc2_weight = fc2.weight().int_repr().detach().cpu().numpy()
    fc2_bias = fc2.bias().detach().cpu().numpy()  # should be int32

    fc1_weight.tofile(prefix + "hweights8.bin")
    fc1_bias.tofile(prefix + "hbias32.bin")
    fc2_weight.tofile(prefix + "outweights8.bin")
    fc2_bias.tofile(prefix + "outbias32.bin")
    print("Exported quantized parameters for eBPF.")


def main(args):
    """script entry-point"""

    params = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        "mnist_data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "mnist_data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=False
    )

    model_fp32 = TinyPerceptron(negative_slope=params.leaky_slope).to(device)
    optimizer = optim.Adam(model_fp32.parameters(), lr=params.learn_rate)

    train(model_fp32, device, train_loader, optimizer, epochs=params.num_epochs)
    train_acc = evaluate(model_fp32, device, train_loader)
    test_acc = evaluate(model_fp32, device, test_loader)
    print(
        f"Float32 Model -> Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%"
    )

    model_fp32.to("cpu")
    model_fp32.eval()

    symmetric_qconfig = torch.ao.quantization.QConfig(
        activation=default_observer,
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    model_fp32.qconfig = symmetric_qconfig

    print("Preparing model for static quantization...")
    model_prepared = quant.prepare(model_fp32, inplace=False)

    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            data = data.to("cpu")
            data = data.view(data.size(0), -1)
            model_prepared(data)
            if i >= 10:
                break

    model_int8 = quant.convert(model_prepared, inplace=False)
    model_int8.to("cpu")
    test_acc_int8 = evaluate(model_int8, torch.device("cpu"), test_loader)
    print(f"Quantized Model -> Test Accuracy: {test_acc_int8:.2f}%")

    export_quantized_parameters(model_int8, prefix="")


if __name__ == "__main__":
    main(sys.argv[1:])

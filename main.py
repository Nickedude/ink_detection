"""Train a small example neural-network to detect ink."""
import os
from pathlib import Path

import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SubVolumeDataset
from data.definitions import IndexRange2D, IndexRange3D
from data.plot import plot_image_with_rectangle, plot_loss
from model.resnet import ResNet


def create_data_loaders(
    path: Path,
    half_width: int,
    indices_to_read: IndexRange3D,
    eval_indices: IndexRange2D,
):
    """Create dataloaders for training, evaluation and inference."""
    batch_size = 32
    train_dataset = SubVolumeDataset(
        path, half_width, indices_to_read, exclude_indices=eval_indices
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
        prefetch_factor=4,
    )

    eval_dataset = SubVolumeDataset(
        path,
        half_width,
        indices_to_read=IndexRange3D(
            x_min=indices_to_read.x_min + eval_indices.x_min,
            x_max=indices_to_read.x_min + eval_indices.x_max,
            y_min=indices_to_read.y_min + eval_indices.y_min,
            y_max=indices_to_read.y_min + eval_indices.y_max,
            z_min=indices_to_read.z_min,
            z_max=indices_to_read.z_max,
        ),
    )
    eval_dataset.subsample(factor=8)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=os.cpu_count(),
        prefetch_factor=4,
        drop_last=False,
    )

    infer_dataset = SubVolumeDataset(path, half_width, indices_to_read=indices_to_read)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, eval_loader, infer_loader


def plot_eval_data(train_dataset: SubVolumeDataset, eval_dataset: SubVolumeDataset):
    """Visualize the evaluation data."""
    width = train_dataset.exclude_indices.y_max - train_dataset.exclude_indices.y_min
    height = train_dataset.exclude_indices.x_max - train_dataset.exclude_indices.x_min

    # Matplotlib uses opposite definition of x/y
    plot_image_with_rectangle(
        image=train_dataset.label.numpy(),
        title="Label mask for training, eval data visualized in red square",
        center=(train_dataset.exclude_indices.y_min, train_dataset.exclude_indices.x_min),
        width=width,
        height=height,
        color="red",
        fill=False,
    )

    plt.imshow(eval_dataset.label.numpy())
    plt.title("Label mask for eval")
    plt.show()


def train(
    device: torch.device, train_loader: DataLoader, eval_loader: DataLoader
) -> torch.nn.Module:
    """Train a neural network on the given data."""
    num_batches = 30_000
    batches_per_epoch = 1000
    train_loader = iter(train_loader)

    model = ResNet().to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = []
    eval_losses = []
    train_losses = []

    for epoch in range(num_batches // batches_per_epoch):
        model.train()
        running_loss = 0.0

        for _ in tqdm(
            range(batches_per_epoch), desc=f"Training epoch {epoch}", unit="batch"
        ):
            optimizer.zero_grad()

            subvolume, label = next(train_loader)
            outputs = model(subvolume.to(device))

            loss = loss_function(outputs, label.to(device))
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        epochs.append(epoch)
        running_loss = running_loss / batches_per_epoch
        train_losses.append(running_loss)

        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for subvolume, label in tqdm(
                eval_loader, desc=f"Eval epoch {epoch}", unit="batch"
            ):
                outputs = model(subvolume.to(device))
                loss = loss_function(outputs, label.to(device))
                running_loss += loss.item()

        eval_losses.append(running_loss / len(eval_loader))
        plot_loss(epochs, train_losses, eval_losses)
        torch.save(model.state_dict(), f"model_weights_{epoch}.pth")

    return model


def infer(
    device: torch.device,
    path: Path,
    half_width: int,
    indices_to_read: IndexRange3D,
    eval_indices: IndexRange2D,
    model: torch.nn.Sequential,
):
    """Run inference on the data and plot the results."""
    batch_size = 32
    eval_dataset = SubVolumeDataset(path, half_width, indices_to_read=indices_to_read)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    output = torch.zeros_like(eval_dataset.mask).float()
    model.eval()

    with torch.no_grad():
        for i, (subvolume, _) in tqdm(
            enumerate(loader), desc="Inferring", total=len(loader)
        ):
            outputs = model(subvolume.to(device))
            for j, value in enumerate(outputs):
                idx = i * batch_size + j
                x, y = eval_dataset.indices[idx].numpy()
                output[x, y] = value

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(output.cpu(), cmap="gray")
    ax2.imshow(eval_dataset.label, cmap="gray")

    def create_rectangle():
        return patches.Rectangle(
            (eval_indices.y_min, eval_indices.x_min),
            width=eval_indices.shape[1],
            height=eval_indices.shape[0],
            color="red",
            fill=False,
        )

    ax1.add_patch(create_rectangle())
    ax2.add_patch(create_rectangle())
    ax1.set_title("Output")
    ax2.set_title("Label")
    plt.show()


def main():
    """Train a small example neural-network to detect ink."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} as device ...")

    path = Path(__file__).parent / "data" / "train" / "1"
    half_width = 31  # Yields 63x63 patches
    indices_to_read = IndexRange3D(
        z_min=27,
        z_max=37,
    )
    eval_indices = IndexRange2D(  # Relative to the read indices
        x_min=4650,
        x_max=5450,
        y_min=1300,
        y_max=2700,
    )
    train_loader, eval_loader, infer_loader = create_data_loaders(
        path, half_width, indices_to_read, eval_indices
    )
    plot_eval_data(train_loader.dataset, eval_loader.dataset)

    print("Training dataset balance:")
    print(f"\tPositive samples: {train_loader.dataset.num_positive_samples}")
    print(f"\tNegative samples: {train_loader.dataset.num_negative_samples}\n")

    print("Eval dataset balance:")
    print(f"\tPositive samples: {eval_loader.dataset.num_positive_samples}")
    print(f"\tNegative samples: {eval_loader.dataset.num_negative_samples}")

    model = train(device, train_loader, eval_loader)
    # model = build().to(device)
    # model.load_state_dict(torch.load("model_weights.pth"))
    infer(device, path, half_width, indices_to_read, eval_indices, model)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()

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
from model.simple import build


def plot_eval_data(dataset: SubVolumeDataset):
    """Visualize the evaluation data."""
    width = dataset.exclude_indices.y_max - dataset.exclude_indices.y_min
    height = dataset.exclude_indices.x_max - dataset.exclude_indices.x_min

    # Matplotlib uses opposite definition of x/y
    plot_image_with_rectangle(
        image=dataset.label.numpy(),
        title="Label mask, eval data visualized in red square",
        center=(dataset.exclude_indices.y_min, dataset.exclude_indices.x_min),
        width=width,
        height=height,
        color="red",
        fill=False,
    )


def train(
    device: torch.device,
    path: Path,
    half_width: int,
    indices_to_read: IndexRange3D,
    eval_indices: IndexRange2D,
) -> torch.nn.Sequential:
    """Train a neural network on the given data."""
    train_dataset = SubVolumeDataset(
        path, half_width, indices_to_read, exclude_indices=eval_indices
    )
    plot_eval_data(train_dataset)

    print(f"Num positive samples: {train_dataset.num_positive_samples}")
    print(f"Num negative samples: {train_dataset.num_negative_samples}")
    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
        prefetch_factor=4,
    )
    num_batches = 10_000

    model = build().to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    running_loss = 0.0
    batches_per_epoch = 100

    losses = []
    epochs = []

    for i, (subvolume, label) in tqdm(
        enumerate(loader), desc="Training", total=num_batches, unit="batch"
    ):
        if i >= num_batches:
            break

        optimizer.zero_grad()
        outputs = model(subvolume.to(device))
        loss = loss_function(outputs, label.to(device))
        loss.backward()
        running_loss += loss.item()

        if (i + 1) % batches_per_epoch == 0:
            epochs.append((i + 1) // batches_per_epoch)
            running_loss = running_loss / batches_per_epoch
            losses.append(running_loss)
            running_loss = 0.0

            plot_loss(epochs, losses)
            torch.save(model.state_dict(), "model_weights.pth")

        optimizer.step()

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
    half_width = 30
    indices_to_read = IndexRange3D(
        x_min=0,
        x_max=1000,
        y_min=2000,
        y_max=5000,
        z_min=27,
        z_max=37,
    )
    eval_indices = IndexRange2D(  # Relative to the read indices
        x_min=0,
        x_max=300,
        y_min=2750,
        y_max=3000,
    )
    model = train(device, path, half_width, indices_to_read, eval_indices)
    # model = build().to(device)
    # model.load_state_dict(torch.load("model_weights.pth"))
    infer(device, path, half_width, indices_to_read, eval_indices, model)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()

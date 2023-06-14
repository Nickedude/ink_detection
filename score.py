"""Module for computing the score of a predicted binary (ink/no-ink) mask."""
import numpy as np
import torch
from torchmetrics import ConfusionMatrix


_EPS = torch.finfo(torch.float32).eps


class ScoreCalculator:

    def __init__(self):
        self._calculator = ConfusionMatrix(task="binary", threshold=0.5)

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """Accumulate predictions and ground truths in a confusion matrix. To be called each batch.

        Args:
            ground_truth: the ground truth binary mask, shape [H, W]
            prediction: the "soft" predictions, a [H, W] tensor with probabilities

        """
        self._calculator.forward(prediction, ground_truth)

    def calculate_score(self) -> float:
        """Calculate the score using the accumulated confusion matrix.

        Should be called at the end of an epoch.
        """
        confusion_matrix = self._calculator.compute()

        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        precision = tp / (tp + fp + _EPS)
        recall = tp / (tp + fn + _EPS)

        return score(precision, recall)


def score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate the score given precision and recall values.

    As per definition:
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview/evaluation

    """
    return ((1 + beta ** 2) * precision * recall) / ((beta ** 2) * precision + recall + _EPS)


def main():
    target = torch.tensor([1, 1, 0, 0])
    preds = torch.tensor([0, 1, 0, 0])
    confmat = ConfusionMatrix(task="binary")
    print(f"Confusion matrix:\n {confmat(preds, target)}")
    print(f"Computed score: {calculate_score(target, preds)}")

    # Plot a 2D image of the score landscape
    num_steps = 10
    step_size = 1.0 / num_steps
    image = np.zeros((num_steps + 1, num_steps + 1))
    eps = np.finfo(np.float32).eps

    for i in range(num_steps + 1):
        for j in range(num_steps + 1):
            image[i, j] = score(i * step_size + eps, j * step_size + eps)

    import matplotlib.pyplot as plt
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.imshow(image)
    plt.title("Score")
    plt.colorbar()
    plt.show()
    plt.close()

    # Plot the score as a function of confidence threshold
    values = np.linspace(start=0.0, stop=1.0, num=100)
    values[0] = eps
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.plot(values, score(values, values[::-1]), label="Score")
    plt.plot(values, values, label="Precision")
    plt.plot(values, values[::-1], label="Recall")
    plt.xlabel("Confidence Threshold")
    plt.legend()
    plt.title("Score")
    plt.show()


if __name__ == "__main__":
    main()

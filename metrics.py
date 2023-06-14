"""A class for tracking metrics throughout a training."""
from collections import defaultdict
import matplotlib.pyplot as plt


class MetricTracker:

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def __call__(self, phase: str, epoch: int, metric: str, value: float):
        self.metrics[metric][phase][epoch].append(value)

    def plot(self):
        for metric in self.metrics:
            self._plot_metric(metric)

    def _plot_metric(self, metric: str):
        for phase in self.metrics[metric]:
            epochs_and_values = sorted(list(self.metrics[metric][phase].items()))  # Sort by epoch
            epochs = [e for e, _ in epochs_and_values]
            values = [v for _, v in epochs_and_values]
            plt.plot(epochs, values, label=phase)

        plt.title(f"{metric} per epoch")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f"{metric}.png")
        plt.close()

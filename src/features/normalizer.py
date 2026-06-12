import numpy as np


class FeatureNormalizer:

    def normalize(self, stack):

        stack = stack.astype(np.float32)

        normalized = np.empty_like(stack)

        stats = {}

        for i in range(stack.shape[-1]):

            channel = stack[:, :, i]

            mean = np.nanmean(channel)

            std = np.nanstd(channel)

            if std < 1e-6:
                std = 1.0

            normalized[:, :, i] = (
                channel - mean
            ) / std

            stats[i] = {
                "mean": float(mean),
                "std": float(std),
            }

        return normalized, stats
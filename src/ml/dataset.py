import numpy as np


class TerrainDataset:

    def __init__(self, context):

        self.context = context

        # Highly recommended for large AOIs
        self.stack = np.load(
            context.features["normalized_stack"],
            mmap_mode="r"
        )

        self.channels = context.features["stack_order"]

    def shape(self):

        return self.stack.shape

    def get_full(self):

        return self.stack

    def get_channel(self, name):

        if name not in self.channels:
            raise ValueError(f"Unknown channel: {name}")

        index = self.channels.index(name)

        return self.stack[:, :, index]

    def get_patch(
        self,
        row,
        col,
        size=256,
    ):

        h, w, _ = self.stack.shape

        # Clamp to image boundaries
        row = max(0, min(row, h - size))
        col = max(0, min(col, w - size))

        return self.stack[
            row:row + size,
            col:col + size,
            :
        ]

    def random_patch(
        self,
        size=256,
    ):

        h, w, _ = self.stack.shape

        if size > h or size > w:
            raise ValueError(
                f"Patch size ({size}) is larger than dataset ({h}, {w})"
            )

        row = np.random.randint(0, h - size + 1)
        col = np.random.randint(0, w - size + 1)

        return self.get_patch(
            row,
            col,
            size,
        )
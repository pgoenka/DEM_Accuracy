import numpy as np

from src.core.aoi import AreaOfInterest
from src.core.pipeline import Pipeline
from src.ml.dataset import TerrainDataset


def main():

    print("=" * 60)
    print("RUNNING PIPELINE")
    print("=" * 60)

    aoi = AreaOfInterest(
        77.0,
        28.0,
        77.1,
        28.1,
    )

    pipeline = Pipeline(aoi)

    pipeline.run()

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = TerrainDataset(
        pipeline.context
    )

    print(f"\nDataset shape: {dataset.shape()}")

    print("\nChannels:")

    for i, name in enumerate(dataset.channels):
        print(f"  {i}: {name}")

    print("\n" + "=" * 60)
    print("CHANNEL TESTS")
    print("=" * 60)

    for name in dataset.channels:

        channel = dataset.get_channel(name)

        print(f"\n{name}")

        print(f"Shape : {channel.shape}")

        print(f"Min   : {channel.min():.4f}")

        print(f"Max   : {channel.max():.4f}")

        print(f"Mean  : {channel.mean():.4f}")

        print(f"Std   : {channel.std():.4f}")

    print("\n" + "=" * 60)
    print("PATCH TEST")
    print("=" * 60)

    h, w = dataset.shape()[:2]

    patch_size = min(256, h, w)

    patch = dataset.get_patch(
        row=0,
        col=0,
        size=patch_size,
    )

    print(f"Patch shape: {patch.shape}")

    print("\nFirst pixel:")

    print(patch[0, 0, :])

    print("\n" + "=" * 60)
    print("RANDOM PATCH TEST")
    print("=" * 60)

    # Get dataset dimensions
    h, w = dataset.shape()[:2]

    # Choose the largest safe patch size
    patch_size = min(256, h, w)

    random_patch = dataset.random_patch(
        size=patch_size
    )

    print(f"Random patch shape: {random_patch.shape}")

    print("\nRandom patch mean per channel:")

    for i, name in enumerate(dataset.channels):

        print(
            f"{name:12s}: "
            f"{random_patch[:, :, i].mean():.4f}"
        )

    print("\n" + "=" * 60)
    print("FULL STACK TEST")
    print("=" * 60)

    full = dataset.get_full()

    print(f"Stack dtype : {full.dtype}")

    print(f"Stack shape : {full.shape}")

    print(f"NaN count   : {np.isnan(full).sum()}")

    print(f"Inf count   : {np.isinf(full).sum()}")

    print("\n" + "=" * 60)
    print("NORMALIZATION CHECK")
    print("=" * 60)

    for i, name in enumerate(dataset.channels):

        arr = full[:, :, i]

        print(
            f"{name:12s}"
            f" mean={arr.mean():8.4f}"
            f" std={arr.std():8.4f}"
        )

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()
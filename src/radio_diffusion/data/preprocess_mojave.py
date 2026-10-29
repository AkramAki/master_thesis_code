"""Preprocess Mojave images for unconditional generation."""

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
from astropy.io import fits
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

def crop_image(path: Path, img_size: int, output_dir: Path) -> None:
    with fits.open(path) as hdul:
        image = torch.from_numpy(
            hdul[0].data[0, 0].astype("float32")
        )
        cropped_image = center_crop(image, img_size).clone()

    output_name = path.name.replace(".icn.fits.gz", ".icn.pt")
    torch.save(cropped_image, output_dir / output_name)


def crop_data(
    input_dir: str,
    img_size: int = 256,
    cpu_cores: int = 1,
) -> None:
    if img_size < 1 or cpu_cores < 1:
        raise ValueError("Image size and CPU worker count must be positive.")

    file_paths = sorted(Path(input_dir).glob("*.fits.gz"))
    if not file_paths:
        raise FileNotFoundError(f"No FITS files found in {input_dir}")

    output_dir = Path(f"/cephfs/users/aaki/data/Mojave_I_{img_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    process_image = partial(
        crop_image,
        img_size=img_size,
        output_dir=output_dir,
    )

    with ProcessPoolExecutor(
        max_workers=cpu_cores,
        initializer=torch.set_num_threads,
        initargs=(1,),
    ) as executor:
        results = executor.map(process_image, file_paths)

        for _ in tqdm(results, total=len(file_paths), desc="Cropping images"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--cpu_cores", type=int, default=1)
    args = parser.parse_args()

    if args.image_size < 1 or args.cpu_cores < 1:
        parser.error("--image_size and --cpu_cores must be positive.")

    crop_data(
        "/cephfs/projects/RADIO/mojave_archive/intensities_fits",
        img_size=args.image_size,
        cpu_cores=args.cpu_cores,
    )
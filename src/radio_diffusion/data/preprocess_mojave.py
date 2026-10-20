"""Preprocess Mojave images for unconditional generation."""

from pathlib import Path

import torch
from astropy.io import fits
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

def crop_data(absolute_path_as_string):
    img_size = 256

    output_dir = Path(f"/cephfs/users/aaki/data/Mojave_I_{img_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = list(Path(absolute_path_as_string).glob("*.fits.gz"))

    for path in tqdm(file_paths, desc="Cropping images", unit="image"):
        with fits.open(path) as hdul:
            image = torch.from_numpy(hdul[0].data[0, 0].astype("float32"))
            cropped_image = center_crop(image[None], img_size)[0]

            output_name = path.name.replace(".icn.fits.gz", ".icn.pt")
            torch.save(cropped_image, output_dir / output_name)


if __name__ == "__main__":
    crop_data("/cephfs/projects/RADIO/mojave_archive/intensities_fits")
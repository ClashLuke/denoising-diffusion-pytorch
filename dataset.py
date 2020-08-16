from io import BytesIO

from PIL import Image
from tensorfn.data import LMDBReader
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.reader = LMDBReader(path, reader="raw")

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        img_bytes = self.reader.get(
            f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
        )

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

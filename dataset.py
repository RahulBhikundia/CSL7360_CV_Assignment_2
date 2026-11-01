import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DUTSDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # verify matching file names
        image_names = [os.path.splitext(f)[0] for f in self.images]
        mask_names = [os.path.splitext(m)[0] for m in self.masks]

        assert image_names == mask_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]
        mask_name = self.masks[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = (mask >= 128).astype("float32")

        image = torch.from_numpy(image) / 255.0 
        image = image.permute(2,0,1)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

if __name__ == '__main__':
    dataset = DUTSDataset(
        "datasets/DUTS-TR/DUTS-TR-Image",
        "datasets/DUTS-TR/DUTS-TR-Mask"
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    images, masks = next(iter(loader))

    print("Batch images:", images.shape)
    print("Batch masks:", masks.shape)
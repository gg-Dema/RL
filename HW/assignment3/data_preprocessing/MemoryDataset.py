from vision_module import VAE
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch


class MemoryCellDataset(Dataset):
    def __init__(self, state_dict_path=None, csv_path=None, img_data_path=None):

        self.vae = VAE()
        self.init_weight(state_dict_path)
        self.init_data_path(csv_path, img_data_path)
        self.len_dataset = len(self.reference_file)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

    def init_weight(self, path_weight):
        if path_weight:
            self.vae.load_state_dict(torch.load(path_weight, map_location='cpu'))
        else:
            self.vae.load_state_dict(torch.load('../model_weights/vae.torch', map_location='cpu'))

    def init_data_path(self, csv_path, img_data_path):
        if csv_path:
            self.reference_file = pd.read_csv(csv_path, sep=',', header=0)
        else:
            self.reference_file = pd.read_csv('../rollouts/data_memory_cell.csv', sep=',', header=0)

        self.img_data_path = '../rollouts/CarRacing_random' if not img_data_path else img_data_path




    def transform_img2hidden(self, idx):  # return z vector [shape = 32]
        image = Image.open(self.img_data_path + '/' + self.reference_file.iloc[idx]['render_path'])
        image = self.transform(image)
        image = torch.unsqueeze(image, 0)
        z, _, _ = self.vae.encode(image)
        return z

    def __getitem__(self, idx, require_predict=True):

        # avoid IndexError
        if idx == self.len_dataset: idx = idx-1


        hidden = self.transform_img2hidden(idx)
        act = self.reference_file.iloc[idx]['action'][1:-1]
        act = np.fromstring(act, sep=' ', dtype=np.float32)
        act = torch.tensor(act).unsqueeze(0)  # unsqueeze : from [3] to [1, 3]
        x = torch.cat((hidden, act), -1)

        if require_predict:
            return {'x': x, 'y': self.__getitem__(idx+1, require_predict=False)}
        else:
            return x


    def __len__(self):
        return len(self.reference_file)




# TEST
if __name__ == '__main__':
    dataset = MemoryCellDataset()
    random_sample = dataset.__getitem__(idx=10)
    print(f"sample[x] shape: {random_sample['x'].shape}")
    print(f"sample[y] shape: {random_sample['y'].shape}")

    #dataloader test:

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=10)
    for i, data in enumerate(dataloader):
        print(data['x'].shape)
        if i == 3:
            break


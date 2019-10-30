import os
import torch.utils.data as data
from . import utils

class CamVid(data.Dataset):

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 loader=utils.h5_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.loader = loader
        self._file_names = self._get_file_names(self.mode.lower())


    def __getitem__(self, index):

        names = self._file_names[index]

        data_path = os.path.join(self.root_dir, names[0])
        label_path = os.path.join(self.root_dir, names[1])

        img, label = self.loader(data_path, label_path)

        if self.mode.lower() == 'train':
            img, label = self.transform(img, label)
        else:
            img = self.transform(img)
            label = self.transform(label)
			

        return img, label

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        split = split_name + '.txt'
        source = os.path.join(self.root_dir, split)

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            img_name, gt_name = self._process_item_names(item)
            file_names.append([img_name, gt_name])

        return file_names

    def _process_item_names(self, item):
        item = item.strip()
        item = item.split('\t')
        img_name = item[0]
        gt_name = item[1]

        return img_name, gt_name

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self._file_names)

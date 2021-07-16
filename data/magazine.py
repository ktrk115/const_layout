from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch_geometric.data import Data

from data.base import BaseDataset


class Magazine(BaseDataset):
    labels = [
        'text',
        'image',
        'headline',
        'text-over-image',
        'headline-over-image',
    ]

    def __init__(self, split='train', transform=None):
        super().__init__('magazine', split, transform)

    def download(self):
        super().download()

    def process(self):
        data_list = []
        ann_dir = Path(self.raw_dir) / 'layoutdata' / 'annotations'
        for xml_path in sorted(ann_dir.glob('*.xml')):
            with xml_path.open() as f:
                root = ET.parse(f).getroot()

            W = float(root.find('size/width').text)
            H = float(root.find('size/height').text)
            name = root.find('filename').text

            elements = root.findall('layout/element')

            boxes = []
            labels = []

            for element in elements:
                # bbox
                px = list(map(float, element.get('polygon_x').split()))
                py = list(map(float, element.get('polygon_y').split()))
                x1, x2 = min(px), max(px)
                y1, y2 = min(py), max(py)
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / W, yc / H,
                     width / W, height / H]
                boxes.append(b)

                # label
                l = element.get('label')
                labels.append(self.label2index[l])

            boxes = torch.tensor(boxes, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)

            data = Data(x=boxes, y=labels)
            data.attr = {
                'name': name,
                'width': W,
                'height': H,
                'has_canvas_element': False,
            }
            data_list.append(data)

        # shuffle with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(data_list), generator=generator)
        data_list = [data_list[i] for i in indices]

        # train 85% / val 5% / test 10%
        N = len(data_list)
        s = [int(N * .85), int(N * .90)]
        torch.save(self.collate(data_list[:s[0]]), self.processed_paths[0])
        torch.save(self.collate(data_list[s[0]:s[1]]), self.processed_paths[1])
        torch.save(self.collate(data_list[s[1]:]), self.processed_paths[2])

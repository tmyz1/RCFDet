# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys

from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class M3FDDataset(CocoDataset):
    """M3FD Dataset for RGB-T detection (COCO format).

    Each sample provides:
        - infrared image (img)
        - visible image (rgb_img)
        - edge image (edge_img, generated from IR)
    """

    METAINFO = {
        'classes': ('People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'),
        'palette': [
            (220, 20, 60), (255, 77, 255), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)],
    }

    def load_data_list(self):
        """Override CocoDataset's method to add RGB + Edge image info."""
        data_list = super().load_data_list()
        for item in data_list:
            img_path = os.path.join(self.data_prefix['img'], item['img_path'])
            rgb_path = img_path.replace('infrared', 'visible')
            item['rgb_img_path'] = rgb_path
            item['edge_img'] = None  # edge 将在 pipeline 中生成
        return data_list

    def parse_data_info(self, raw_data_info):
        """Add RGB path and IR path into data_info for pipeline use."""
        data_info = super().parse_data_info(raw_data_info)
        if 'infrared' in data_info['img_path']:
            data_info['rgb_img_path'] = data_info['img_path'].replace('infrared', 'visible')
        return data_info




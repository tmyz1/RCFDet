import os

from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class M3FDFusionDataset(CocoDataset):
    """M3FD COCO dataset wrapper for image-level fusion results.

    The original M3FD annotations use file names such as ``00000.png``, while
    FSATFusion saves fused images as ``00000_FSATFusion.png``. This dataset
    keeps the original annotations and only redirects image paths.
    """

    METAINFO = {
        'classes': ('People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'),
        'palette': [
            (220, 20, 60), (255, 77, 255), (0, 0, 142),
            (0, 0, 230), (106, 0, 228), (0, 60, 100)
        ],
    }

    def __init__(self, fusion_suffix='_FSATFusion', **kwargs):
        self.fusion_suffix = fusion_suffix
        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)
        img_path = data_info['img_path']
        stem, ext = os.path.splitext(img_path)
        data_info['img_path'] = f'{stem}{self.fusion_suffix}{ext}'
        return data_info

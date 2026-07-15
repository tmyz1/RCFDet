import os

import cv2
import numpy as np
from mmdet.registry import MODELS
import torch.nn.functional as F
import math
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor

from mmdetection.mmdet.structures.bbox import HorizontalBoxes


@MODELS.register_module()
class RGBTDataPreprocessor(DetDataPreprocessor):

    def __init__(self, *args, visualize=False, save_dir='preprocessor_vis', **kwargs):
        super().__init__(*args, **kwargs)
        self.visualize = visualize
        self.save_dir = save_dir
        self.batch_count = 0

    def forward(self, data, training=False):
        data = self.cast_data(data)
        _batch_inputs = data['inputs']

        # 现在应该是 (rgb, ir, a)
        assert isinstance(_batch_inputs, (list, tuple)) and len(_batch_inputs) == 3, \
            f'Expected (rgb, ir, a), but got {type(_batch_inputs)}'

        rgb, ir, a = _batch_inputs

        def _process_image(modality):
            """Only process image-like tensors: [B,C,H,W]."""
            if modality.dim() == 4:
                modality = modality.float()
                if self._enable_normalize:
                    if modality.shape[1] == 3:
                        modality = (modality - self.mean) / self.std
                    elif modality.shape[1] == 1:
                        modality = (modality - self.mean.mean()) / self.std.mean()

                h, w = modality.shape[2:]
                target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h, pad_w = target_h - h, target_w - w

                modality = F.pad(modality, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
                return modality, pad_h, pad_w
            else:
                return modality, 0, 0  # for scalar a

        # 处理两路图像
        rgb, pad_h, pad_w = _process_image(rgb)
        ir, _, _ = _process_image(ir)

        # a 不做任何处理，只保持 tensor shape=[B] 或 [B,1]
        a = a.float()

        # 更新 inputs
        data['inputs'] = (rgb, ir, a)

        # 填入 shapes
        if 'data_samples' in data and data['data_samples'] is not None:
            batch_input_shape = rgb.shape[-2:]
            for data_sample in data['data_samples']:
                data_sample.set_metainfo(dict(batch_input_shape=batch_input_shape))

        # 可视化（仅 rgb + bbox）
        if self.visualize:
            visualize_preprocessed_data(
                rgb=rgb,
                ir=ir,
                edge= None,
                data_samples=data.get('data_samples'),
                mean=self.mean if self._enable_normalize else None,
                std=self.std if self._enable_normalize else None,
                pad_h=pad_h,
                pad_w=pad_w,
                save_dir=self.save_dir,
                batch_idx=self.batch_count
            )
            self.batch_count += 1

        return data



def visualize_preprocessed_data(
        rgb, ir, edge, data_samples,
        mean, std, pad_h, pad_w,
        save_dir='preprocessor_vis',
        batch_idx=0
):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = rgb.shape[0]

    for i in range(batch_size):
        rgb_img = rgb[i].cpu().detach().numpy()  # [C, H, W]
        rgb_img = np.transpose(rgb_img, (1, 2, 0))  # [H, W, 3]

        if mean is not None and std is not None:
            mean_np = mean.cpu().numpy().squeeze()  # 确保形状为(3,)
            std_np = std.cpu().numpy().squeeze()
            rgb_img = rgb_img * std_np + mean_np

        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 转为BGR（3通道）

        if data_samples is not None and i < len(data_samples):
            data_sample = data_samples[i]
            bboxes = data_sample.gt_instances.bboxes
            if isinstance(bboxes, HorizontalBoxes):
                bboxes = bboxes.tensor.cpu().numpy()
            for box in bboxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        assert rgb_img.shape[-1] == 3, f"RGB通道数错误: {rgb_img.shape[-1]}"

        # -------------------------- 保存图像 --------------------------
        base_name = f"batch_{batch_idx}_sample_{i}"
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_rgb.jpg"), rgb_img)
        print(f"[Vis] 已保存预处理后图像: {save_dir}/{base_name}_*.jpg")
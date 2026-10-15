import csv
import math
from pathlib import Path

import cv2
import torch
from mmcv import BaseTransform, to_tensor
from mmdet.registry import TRANSFORMS
import mmcv
from mmcv.image import imread
import numpy as np
from mmengine.structures import InstanceData, PixelData
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class LoadRGBTImageFromFile(BaseTransform):
    """Load pre-enhanced RGB/IR images and the image-level RGB weight."""

    def __init__(
        self,
        weights_file,
        to_float32=True,
        color_type='color',
        imdecode_backend='cv2',
    ):
        super().__init__()
        self.weights_file = Path(weights_file)
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self._rgb_weights = None

    def _load_weights(self):
        weights = {}
        with self.weights_file.open('r', encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file, delimiter='\t')
            required = {'image', 'rgb_weight', 'ir_weight'}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f'Invalid weight columns in {self.weights_file}')
            for row in reader:
                name = row['image'].replace('\\', '/')
                if not name or name in weights:
                    raise ValueError(f'Missing or duplicate image name in {self.weights_file}: {name}')
                rgb_weight = float(row['rgb_weight'])
                ir_weight = float(row['ir_weight'])
                if (not math.isfinite(rgb_weight) or not math.isfinite(ir_weight)
                        or not 0.0 <= rgb_weight <= 1.0
                        or not 0.0 <= ir_weight <= 1.0
                        or abs(rgb_weight + ir_weight - 1.0) > 1e-5):
                    raise ValueError(f'Invalid modality weights for {name}')
                weights[name] = rgb_weight
        if not weights:
            raise ValueError(f'No image weights found in {self.weights_file}')
        self._rgb_weights = weights

    def transform(self, results: dict):
        if self._rgb_weights is None:
            self._load_weights()
        image_root = self.weights_file.parent.resolve()
        ir_path = Path(results['img_path']).resolve()
        try:
            name = ir_path.relative_to(image_root / 'infrared').as_posix()
        except ValueError as exc:
            raise ValueError(
                f'IR image must be under {image_root / "infrared"}: {ir_path}'
            ) from exc
        if name not in self._rgb_weights:
            raise KeyError(f'No RGB weight for {name} in {self.weights_file}')
        rgb_path = image_root / 'visible' / name

        ir = imread(str(ir_path), flag=self.color_type, backend=self.imdecode_backend)
        rgb = imread(str(rgb_path), flag=self.color_type, backend=self.imdecode_backend)
        if ir is None or rgb is None:
            raise FileNotFoundError(f'Failed to read image {ir_path} or {rgb_path}')
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.to_float32:
            ir = ir.astype(np.float32)
            rgb = rgb.astype(np.float32)

        results['img'] = ir
        results['rgb_img'] = rgb
        results.pop('edge_img', None)
        results['rgb_img_path'] = str(rgb_path)
        results['a'] = self._rgb_weights[name]
        results['img_shape'] = ir.shape[:2]
        results['ori_shape'] = ir.shape[:2]
        return results


@TRANSFORMS.register_module()
class ResizeRGBT(BaseTransform):
    def __init__(self, scale=(224, 224), keep_ratio=False):
        super().__init__()
        self.scale = scale
        self.keep_ratio = keep_ratio

    def transform(self, results):
        # 记录原始尺寸
        ori_h, ori_w = results['img'].shape[:2]
        new_w, new_h = self.scale
        w_scale = new_w / ori_w
        h_scale = new_h / ori_h

        # resize 所有模态
        for key in ['img', 'rgb_img', 'edge_img', 'fusion_img', 'fusion_alpha']:
            if key in results:
                #results[key] = mmcv.imresize(results[key], (new_w, new_h))
                results[key] = cv2.resize(results[key], (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # bbox 同步缩放
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']

            # 取出内部 Tensor
            if hasattr(bboxes, 'tensor'):
                bboxes = bboxes.tensor

            # 计算缩放因子
            scale_factor = torch.tensor([w_scale, h_scale, w_scale, h_scale], dtype=bboxes.dtype, device=bboxes.device)

            # 缩放坐标
            bboxes = bboxes * scale_factor

            # 再包回 HorizontalBoxes
            results['gt_bboxes'] = type(results['gt_bboxes'])(bboxes)
            results['scale_factor'] = np.array([w_scale, h_scale], dtype=np.float32)
        else:
            results['scale_factor'] = np.array([w_scale, h_scale], dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['ori_shape'] = (ori_h, ori_w)
        return results

@TRANSFORMS.register_module()
class RandomFlipRGBT(BaseTransform):
    """Randomly flip IR, RGB, and Edge images synchronously."""

    def __init__(self, prob=0.5, direction='horizontal'):
        self.prob = prob
        self.direction = direction

    def transform(self, results):
        if np.random.rand() < self.prob:
            for key in ['img', 'rgb_img', 'edge_img', 'fusion_img', 'fusion_alpha']:
                if key in results:
                    results[key] = mmcv.imflip(results[key], direction=self.direction)

            if 'gt_bboxes' in results:
                bboxes = results['gt_bboxes']
                if hasattr(bboxes, 'tensor'):
                    boxes_tensor = bboxes.tensor.clone()
                else:
                    boxes_tensor = bboxes.clone()

                h, w = results['img_shape']
                if self.direction == 'horizontal':
                    # [x1, y1, x2, y2]
                    boxes_tensor[:, [0, 2]] = w - boxes_tensor[:, [2, 0]]
                results['gt_bboxes'] = type(results['gt_bboxes'])(boxes_tensor)

        return results

@TRANSFORMS.register_module()
class RandomVerticalFlipRGBT(BaseTransform):
    """Randomly flip IR, RGB, and Edge images vertically and synchronously."""

    def __init__(self, prob=0.5):
        self.prob = prob
        self.direction = 'vertical'

    def transform(self, results):
        if np.random.rand() < self.prob:
            # 翻转图像
            for key in ['img', 'rgb_img', 'edge_img', 'fusion_img', 'fusion_alpha']:
                if key in results:
                    results[key] = mmcv.imflip(results[key], direction=self.direction)

            # 翻转 bbox
            if 'gt_bboxes' in results:
                bboxes = results['gt_bboxes']

                # 强制转 tensor (兼容 BaseBoxes / Tensor)
                if hasattr(bboxes, 'tensor'):
                    boxes_tensor = bboxes.tensor.clone()
                else:
                    boxes_tensor = bboxes.clone()

                h, w = results['img_shape']

                # 垂直翻转：y1,y2 坐标变化  [x1, y1, x2, y2]
                boxes_tensor[:, [1, 3]] = h - boxes_tensor[:, [3, 1]]

                # 用原类型返回
                results['gt_bboxes'] = type(results['gt_bboxes'])(boxes_tensor)
        return results

@TRANSFORMS.register_module()
class RandomBlurRGBT(BaseTransform):
    """Random Gaussian Blur for RGB, IR, Edge images (size unchanged)."""

    def __init__(self, prob=0.5, ksize=5):
        self.prob = prob
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1  # kernel 必须是奇数

    def transform(self, results):
        if np.random.rand() < self.prob:
            for key in ['img', 'rgb_img', 'edge_img', 'fusion_img']:
                if key in results:
                    results[key] = cv2.GaussianBlur(results[key], (self.ksize, self.ksize), 0)
        return results

@TRANSFORMS.register_module()
class RandomCutoutRGBT(BaseTransform):
    """Random rectangle cutout (erase) for RGB, IR, Edge images."""

    def __init__(self, prob=0.5, max_size=0.3):
        """
        max_size: 最大遮挡比例，例如 0.3 表示遮挡区域最大为 30% 高宽
        """
        self.prob = prob
        self.max_size = max_size

    def transform(self, results):
        if np.random.rand() < self.prob:
            h, w = results['img_shape']

            cut_w = int(np.random.uniform(0.1, self.max_size) * w)
            cut_h = int(np.random.uniform(0.1, self.max_size) * h)

            x1 = np.random.randint(0, w - cut_w)
            y1 = np.random.randint(0, h - cut_h)

            for key in ['img', 'rgb_img', 'ir_img', 'edge_img', 'fusion_img', 'fusion_alpha']:
                if key in results:
                    img = results[key]
                    img[y1:y1+cut_h, x1:x1+cut_w] = 0
                    results[key] = img

        return results

@TRANSFORMS.register_module()
class RandomBrightnessContrastRGBT(BaseTransform):
    """Random brightness & contrast adjustment."""

    def __init__(self, prob=0.5, brightness_range=0.2, contrast_range=0.2):
        self.prob = prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def transform(self, results):
        if np.random.rand() < self.prob:

            # 亮度 (偏移)
            b = np.random.uniform(-self.brightness_range, self.brightness_range) * 255

            # 对比度 (缩放)
            c = 1 + np.random.uniform(-self.contrast_range, self.contrast_range)

            for key in ['img', 'rgb_img', 'ir_img', 'edge_img', 'fusion_img']:
                if key in results:
                    img = results[key].astype(np.float32)
                    img = img * c + b
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    results[key] = img

        return results

@TRANSFORMS.register_module()
class PackDetRGBTInputs(BaseTransform):
    """
    打包 RGB + IR + Edge 三模态输入的版本，
    用于目标检测模型输入。

    预期输入（来自 pipeline）:
        results = {
            'img': IR 图像,          # np.ndarray, H×W×3
            'rgb_img': RGB 图像,     # np.ndarray, H×W×3
            'edge_img': Edge 图像,   # np.ndarray, H×W×1 或 H×W
            'gt_bboxes': HorizontalBoxes 或 np.ndarray(N, 4),
            'gt_labels': np.ndarray(N,),
            'gt_ignore_flags': np.ndarray(N,) (可选),
            'gt_masks': BaseBoxes 或 np.ndarray,
            'img_shape': (H, W),
            ...
        }

    输出:
        {
            'inputs': {
                'rgb': Tensor(C, H, W),
                'ir': Tensor(C, H, W),
                'edge': Tensor(1, H, W)
            },
            'data_samples': DetDataSample()
        }
    """

    def __init__(self, meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')):
        self.meta_keys = meta_keys

    def transform(self, results: dict):
        packed = dict()
        # 处理 IR 图像
        ir_img = results['img']
        if ir_img.ndim == 2:
            ir_img = np.expand_dims(ir_img, -1)
        ir_img = np.ascontiguousarray(ir_img)  # ✅ 确保正 stride
        ir_img = to_tensor(ir_img).permute(2, 0, 1).contiguous()

        # 处理 RGB 图像
        rgb_img = results.get('rgb_img', None)
        if rgb_img is not None:
            rgb_img = np.ascontiguousarray(rgb_img)
            rgb_img = to_tensor(rgb_img).permute(2, 0, 1).contiguous()


        a = torch.tensor(results['a'], dtype=torch.float32)

        packed['inputs']=(rgb_img, ir_img, a)

        # 处理标注信息
        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        # 处理 ignore 标志
        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]
        else:
            valid_idx = None

        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            if isinstance(bboxes, HorizontalBoxes):
                bboxes = bboxes.tensor
            elif isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes)
            if valid_idx is not None:
                instance_data.bboxes = bboxes[valid_idx]
                ignore_instance_data.bboxes = bboxes[ignore_idx]
            else:
                instance_data.bboxes = bboxes

        if 'gt_bboxes_labels' in results:
            labels = results['gt_bboxes_labels']
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            if valid_idx is not None:
                instance_data.labels = labels[valid_idx]
                ignore_instance_data.labels = labels[ignore_idx]
            else:
                instance_data.labels = labels

        if 'gt_masks' in results:
            masks = results['gt_masks']
            if valid_idx is not None:
                instance_data.masks = masks[valid_idx]
                ignore_instance_data.masks = masks[ignore_idx]
            else:
                instance_data.masks = masks

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores'])
            )
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                gt_sem_seg_data.set_metainfo(dict(ignore_index=results['ignore_index']))
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed['data_samples'] = data_sample

        #visualize_bboxes(results, prefix='pack_debug')

        return dict(inputs=packed['inputs'],
    data_samples=packed['data_samples']
)

def visualize_bboxes(results, prefix=''):
    img = results['rgb_img'] if 'rgb_img' in results else results['img']
    img_show = img.copy()
    if isinstance(img_show, torch.Tensor):
        img_show = img_show.permute(1, 2, 0).cpu().numpy()
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
    img_show = (img_show * 255).astype(np.uint8)
    img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

    bboxes = results.get('gt_bboxes', None)
    if bboxes is not None:
        bboxes = bboxes.numpy() if hasattr(bboxes, 'numpy') else np.array(bboxes)
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

    save_path = f'temp_debug_{prefix}.jpg'
    cv2.imwrite(save_path, img_show)
    print(f"[DEBUG] Saved visualization to {save_path}, shape={img_show.shape}")


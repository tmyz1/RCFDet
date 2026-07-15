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
    """Load IR + RGB + Edge images from file."""
    def __init__(
        self,
        to_float32=True,
        color_type='color',
        imdecode_backend='cv2',
        fft_sigma=10.0,
        fft_pad=64,
        global_max_shift=24,
        global_min_ncc=0.18,
        global_min_phase_response=0.03,
        fusion_block=16,
        fusion_search_radius=5,
        fusion_search_step=1,
        fusion_min_ncc=0.18,
        fusion_response_margin=0.08,
        fusion_ncc_weight=0.10,
        fusion_detail_weight=0.45,
        fusion_ir_luma_weight=0.35,
    ):
        super(LoadRGBTImageFromFile, self).__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.fft_sigma = fft_sigma
        self.fft_pad = fft_pad
        self.global_max_shift = global_max_shift
        self.global_min_ncc = global_min_ncc
        self.global_min_phase_response = global_min_phase_response
        self.fusion_block = fusion_block
        self.fusion_search_radius = fusion_search_radius
        self.fusion_search_step = fusion_search_step
        self.fusion_min_ncc = fusion_min_ncc
        self.fusion_response_margin = fusion_response_margin
        self.fusion_ncc_weight = fusion_ncc_weight
        self.fusion_detail_weight = fusion_detail_weight
        self.fusion_ir_luma_weight = fusion_ir_luma_weight

    def transform(self, results: dict):
        ir_path = results['img_path']
        rgb_path = results.get('rgb_img_path', ir_path.replace('infrared', 'visible'))

        ir = imread(ir_path, flag=self.color_type, backend=self.imdecode_backend)
        rgb = imread(rgb_path, flag=self.color_type, backend=self.imdecode_backend)

        if ir is None or rgb is None:
            raise FileNotFoundError(f'Failed to read image {ir_path} or {rgb_path}')

        if ir.shape[:2] != rgb.shape[:2]:
            ir = self._resize_like(ir, rgb)

        fusion = self._fuse_gaussian_fft(rgb, ir)
        a = fusion['selected_ratio']
        use_pixel_fusion = 0.0 < a < 1.0
        rgb_for_model = fusion['fused_color'] if use_pixel_fusion else rgb

        if self.to_float32:
            ir = ir.astype(np.float32)
            rgb = rgb.astype(np.float32)
            rgb_for_model = rgb_for_model.astype(np.float32)

        results['img'] = ir
        results['rgb_img'] = rgb_for_model
        results['edge_img'] = fusion['fused_feature']
        results['fusion_img'] = fusion['fused_color']
        results['fusion_alpha'] = fusion['alpha']
        results['a'] = a
        results['use_pixel_fusion'] = use_pixel_fusion
        results['img_shape'] = ir.shape[:2]
        results['ori_shape'] = ir.shape[:2]
        return results

    def _to_uint8_image(self, img):
        arr = np.nan_to_num(np.asarray(img)).astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _normalize_u8(self, x):
        x = np.asarray(x, dtype=np.float32)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        return np.uint8((x - mn) / (mx - mn) * 255.0)

    def _normalize_float(self, x):
        x = np.asarray(x, dtype=np.float32)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    def _to_gray(self, img):
        img = self._to_uint8_image(img)
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _resize_like(self, src, ref):
        h, w = ref.shape[:2]
        if src.shape[:2] == (h, w):
            return src
        return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)

    def _warp_image(self, img, dx, dy, border=cv2.BORDER_REFLECT_101):
        h, w = img.shape[:2]
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            img,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=border,
        )

    def fft_H(self, img, sigma=None, pad=None):
        """Gaussian FFT high-pass filtering with reflect padding."""
        sigma = self.fft_sigma if sigma is None else sigma
        pad = self.fft_pad if pad is None else pad
        gray = self._to_uint8_image(img).astype(np.float32) / 255.0
        if pad > 0:
            src = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        else:
            src = gray

        rows, cols = src.shape
        dft = cv2.dft(src, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows)
        v = np.arange(cols)
        vv, uu = np.meshgrid(v, u)
        dist2 = (uu - crow) ** 2 + (vv - ccol) ** 2
        mask = 1.0 - np.exp(-dist2 / (2.0 * sigma * sigma))
        mask = np.repeat(mask[:, :, None], 2, axis=2)

        filtered = dft_shift * mask
        inv = cv2.idft(np.fft.ifftshift(filtered))
        mag = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

        if pad > 0:
            mag = mag[pad:-pad, pad:-pad]
        return self._normalize_u8(mag)

    def _ncc(self, a, b):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        return float(np.sum(a * b) / denom)

    def _estimate_global_shift(self, rgb_feature, ir_feature):
        rgb = self._normalize_float(cv2.GaussianBlur(rgb_feature, (0, 0), 1.2))
        ir = self._normalize_float(cv2.GaussianBlur(ir_feature, (0, 0), 1.2))

        shift, response = cv2.phaseCorrelate(rgb.astype(np.float32), ir.astype(np.float32))
        sx, sy = float(shift[0]), float(shift[1])
        sx = float(np.clip(sx, -self.global_max_shift, self.global_max_shift))
        sy = float(np.clip(sy, -self.global_max_shift, self.global_max_shift))

        candidates = [(0.0, 0.0), (sx, sy), (-sx, -sy)]
        best = (0.0, 0.0)
        best_score = -1e9
        for dx, dy in candidates:
            warped = self._warp_image(ir, dx, dy)
            score = self._ncc(rgb, warped)
            if score > best_score:
                best_score = score
                best = (dx, dy)
        return best[0], best[1], response, best_score

    def _build_local_flow(self, rgb_feature, ir_feature, alpha_scale=0.45):
        rgb = self._normalize_float(rgb_feature)
        ir = self._normalize_float(ir_feature)
        h, w = rgb.shape

        alpha = np.zeros((h, w), dtype=np.float32)
        dx_map = np.zeros((h, w), dtype=np.float32)
        dy_map = np.zeros((h, w), dtype=np.float32)
        selected = 0
        rejected = 0
        ncc_values = []
        block = self.fusion_block
        search_radius = self.fusion_search_radius
        search_step = max(1, self.fusion_search_step)

        for y in range(0, h, block):
            for x in range(0, w, block):
                y2 = min(y + block, h)
                x2 = min(x + block, w)
                bh, bw = y2 - y, x2 - x
                rgb_block = rgb[y:y2, x:x2]
                rgb_score = float(np.mean(rgb_block))

                y_start = max(0, y - search_radius)
                y_end = min(h - bh, y + search_radius)
                x_start = max(0, x - search_radius)
                x_end = min(w - bw, x + search_radius)
                yy_list = list(range(y_start, y_end + 1, search_step))
                xx_list = list(range(x_start, x_end + 1, search_step))
                if y not in yy_list:
                    yy_list.append(y)
                if x not in xx_list:
                    xx_list.append(x)

                best_score = -1e9
                best_ncc = -1.0
                best_x = x
                best_y = y
                for yy in sorted(set(yy_list)):
                    for xx in sorted(set(xx_list)):
                        ir_block = ir[yy:yy + bh, xx:xx + bw]
                        cur_ncc = self._ncc(rgb_block, ir_block)
                        if cur_ncc < self.fusion_min_ncc:
                            continue
                        score = float(np.mean(ir_block)) + self.fusion_ncc_weight * cur_ncc
                        if score > best_score:
                            best_score = score
                            best_ncc = cur_ncc
                            best_x = xx
                            best_y = yy

                advantage = best_score - rgb_score
                if advantage > self.fusion_response_margin and best_ncc >= self.fusion_min_ncc:
                    cur_alpha = np.clip((advantage / alpha_scale) * best_ncc, 0.0, 1.0)
                    alpha[y:y2, x:x2] = cur_alpha
                    dx_map[y:y2, x:x2] = best_x - x
                    dy_map[y:y2, x:x2] = best_y - y
                    selected += 1
                    ncc_values.append(best_ncc)
                else:
                    rejected += 1

        if np.any(alpha > 0):
            dx_map = cv2.medianBlur(dx_map.astype(np.float32), 5)
            dy_map = cv2.medianBlur(dy_map.astype(np.float32), 5)
            alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
            alpha = np.clip(alpha, 0.0, 1.0)

        stats = {
            'selected_blocks': selected,
            'rejected_blocks': rejected,
            'selected_ratio': selected / max(selected + rejected, 1),
            'mean_local_ncc': float(np.mean(ncc_values)) if ncc_values else 0.0,
        }
        return dx_map, dy_map, alpha, stats

    def _remap_with_flow(self, img, dx_map, dy_map):
        h, w = img.shape[:2]
        xs, ys = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = xs + dx_map.astype(np.float32)
        map_y = ys + dy_map.astype(np.float32)
        return cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _fuse_gaussian_fft(self, rgb_img, ir_img):
        rgb_u8 = self._to_uint8_image(rgb_img)
        ir_u8 = self._resize_like(self._to_uint8_image(ir_img), rgb_u8)
        rgb_gray = self._to_gray(rgb_u8)
        ir_gray = self._to_gray(ir_u8)

        rgb_feature = self.fft_H(rgb_gray)
        ir_feature = self.fft_H(ir_gray)
        gdx, gdy, phase_response, global_ncc = self._estimate_global_shift(
            rgb_feature, ir_feature
        )
        if (phase_response < self.global_min_phase_response
                or global_ncc < self.global_min_ncc):
            gdx, gdy = 0.0, 0.0

        ir_feature_global = self._warp_image(ir_feature, gdx, gdy)
        ir_gray_global = self._warp_image(ir_gray, gdx, gdy)
        dx_map, dy_map, alpha, stats = self._build_local_flow(
            rgb_feature, ir_feature_global
        )
        ir_feature_warped = self._remap_with_flow(ir_feature_global, dx_map, dy_map)
        ir_gray_warped = self._remap_with_flow(ir_gray_global, dx_map, dy_map)

        fused_float = (
            (1.0 - alpha) * self._normalize_float(rgb_feature)
            + alpha * self._normalize_float(ir_feature_warped)
        )
        fused_feature = self._normalize_u8(fused_float)

        detail = cv2.GaussianBlur(fused_feature, (0, 0), 0.6)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        rgb_v = hsv[:, :, 2]
        ir_hot = self._normalize_float(ir_gray_warped)
        hot_start = float(np.percentile(ir_hot, 72))
        hot_end = float(np.percentile(ir_hot, 98))
        ir_hot = np.clip((ir_hot - hot_start) / (hot_end - hot_start + 1e-8), 0.0, 1.0)
        alpha_body = cv2.GaussianBlur(alpha, (0, 0), 6.0)
        luma_alpha = np.clip(np.maximum(alpha, alpha_body * ir_hot), 0.0, 1.0)
        ir_positive_luma = np.maximum(ir_gray_warped.astype(np.float32) - rgb_v, 0.0)
        fused_v = (
            rgb_v
            + self.fusion_detail_weight * detail.astype(np.float32)
            + self.fusion_ir_luma_weight * luma_alpha * ir_positive_luma
        )
        hsv[:, :, 2] = np.clip(fused_v, 0, 255)
        fused_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return {
            'rgb_feature': rgb_feature,
            'ir_feature': ir_feature,
            'fused_feature': fused_feature,
            'fused_color': fused_color,
            'alpha': alpha.astype(np.float32),
            'selected_ratio': float(stats['selected_ratio']),
            'global_dx': float(gdx),
            'global_dy': float(gdy),
            'global_ncc': float(global_ncc),
        }

    def region_select_fft(self, rgb_fft, ir_fft, block=10):
        h, w = rgb_fft.shape
        add = np.zeros_like(rgb_fft, dtype=np.uint8)
        cnt1 = 0
        cnt2 = 0
        for y in range(0, h, block):
            for x in range(0, w, block):
                y2 = min(y + block, h)
                x2 = min(x + block, w)
                mean_1 = np.mean(rgb_fft[y:y2, x:x2])
                mean_2 = np.mean(ir_fft[y:y2, x:x2])
                if mean_1 > mean_2:
                    add[y:y2, x:x2] = rgb_fft[y:y2, x:x2]
                    cnt1 += 1
                else:
                    add[y:y2, x:x2] = ir_fft[y:y2, x:x2]
                    cnt2 += 1
        return add, cnt1 - cnt2

    def get_score(self, img, number, block):
        h, w = img.shape
        total = (h // block) * (w // block)
        num = abs(number)
        if number > 0:
            score = (num + total) / (total * 2)
        else:
            score = num / (total * 2)
        if score < 0.2:
            score = 0
        if score > 0.8:
            score = 1
        return score


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


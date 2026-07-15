# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
import time
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FPSCalculateHook(Hook):
    """端到端推理速度统计 Hook（兼容 Windows + MMDet 3.x）"""
    priority = 'VERY_LOW'

    def before_run(self, runner) -> None:
        self.start_time = time.time()
        self.total_samples = 0
        self.batch_count = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runner.logger.info("[FPS Hook] Start timing...")

    def after_iter(self, runner) -> None:
        # MMDet 3.x: runner.data_batch 是 dict，包含 'inputs' 和 'data_samples'
        batch = getattr(runner, 'data_batch', None)
        if batch is None:
            return

        # 方法1: 从 data_samples 获取（推荐）
        data_samples = batch.get('data_samples', [])
        if isinstance(data_samples, (list, tuple)):
            batch_size = len(data_samples)
        else:
            # 方法2: 从 inputs 获取（备选）
            inputs = batch.get('inputs', None)
            if inputs is not None:
                batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else 1
            else:
                batch_size = 1  # 保底

        self.total_samples += batch_size
        self.batch_count += 1

        # 每 50 个 batch 打印一次进度（可选）
        if self.batch_count % 50 == 0:
            elapsed = time.time() - self.start_time
            cur_fps = self.total_samples / elapsed if elapsed > 0 else 0
            runner.logger.info(f"[FPS] Processed {self.total_samples} imgs, {cur_fps:.2f} fps...")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def after_run(self, runner) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - self.start_time

        # 避免除零
        if self.total_samples > 0 and elapsed > 0:
            fps = self.total_samples / elapsed
            latency = (elapsed / self.total_samples) * 1000
        else:
            fps = 0
            latency = 0

        # 使用 ASCII 字符，避免 Windows GBK 编码错误
        runner.logger.info("\n" + "=" * 60)
        runner.logger.info(" [FPS Statistics] End-to-end inference speed")
        runner.logger.info(f"   Total images  : {self.total_samples}")
        runner.logger.info(f"   Total time    : {elapsed:.2f} seconds")
        runner.logger.info(f"   Avg latency   : {latency:.2f} ms/img")
        runner.logger.info(f"   >> FPS        : {fps:.2f} img/s")
        runner.logger.info("=" * 60 + "\n")


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline


    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # ================= 新增：端到端推理计时 =================
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    runner.logger.info("[FPS] Start end-to-end inference timing...")
    # ========================================================
    # start testing
    runner.test()

    # ================= 新增：计算并打印 FPS =================
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed = end_time - start_time

    # 从测试集配置中获取图片总数（避免依赖 batch 统计）
    test_dataset = cfg.test_dataloader.dataset
    if hasattr(test_dataset, 'ann_file') and osp.exists(test_dataset.ann_file):
        import json
        with open(test_dataset.ann_file, 'r') as f:
            ann_data = json.load(f)
        total_images = len(ann_data.get('images', []))
    else:
        # 兜底：用 dataloader 长度估算
        total_images = len(runner.test_dataloader) * cfg.test_dataloader.get('batch_size', 1)

    if total_images > 0 and elapsed > 0:
        fps = total_images / elapsed
        latency = (elapsed / total_images) * 1000
    else:
        fps = 0
        latency = 0

    runner.logger.info("\n" + "=" * 60)
    runner.logger.info(" [FPS Statistics] End-to-end inference speed")
    runner.logger.info(f"   Total images  : {total_images}")
    runner.logger.info(f"   Total time    : {elapsed:.2f} seconds")
    runner.logger.info(f"   Avg latency   : {latency:.2f} ms/img")
    runner.logger.info(f"   >> FPS        : {fps:.2f} img/s")
    runner.logger.info("=" * 60 + "\n")
    # ========================================================


if __name__ == '__main__':
    main()

import os
import time
import torch
import torch.cuda as cuda
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from thop import profile, clever_format


def analyze_model(config_path, checkpoint_path=None,
                  input_size=(384, 384), channels=3,
                  warmup=50, test_iter=200, device='cuda:0'):
    print(f"\n{'=' * 60}")
    print(f"📊 MMDet 3.x 模型综合分析工具")
    print(f"{'=' * 60}\n")

    # 1. 加载配置 & 构建模型
    print("📂 加载配置...")
    cfg = Config.fromfile(config_path)
    cfg.load_from = None  # 统计不需要加载权重，加快速度
    model = MODELS.build(cfg.model)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"⏳ 加载权重: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)

    model = model.to(device)
    model.eval()

    # 2. 参数量统计
    print(f"\n{'─' * 60}")
    print(" 参数量统计")
    print(f"{'─' * 60}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数量: {total_params / 1e6:.2f} M")
    print(f"🔧 可训练:   {trainable / 1e6:.2f} M ({trainable / total_params * 100:.1f}%)")

    # 3. FLOPs 统计（安全包装，避开 data_samples 冲突）
    print(f"\n{'─' * 60}")
    print("⚡ FLOPs 统计")
    print(f"{'─' * 60}")

    class MMDetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # MMDet 3.x 要求输入为 dict，内部包含 data_samples
            data_samples = [DetDataSample()]
            return self.model(dict(inputs=x, data_samples=data_samples))

    wrapper = MMDetWrapper(model)
    dummy = torch.randn(1, channels, *input_size, device=device)

    print(f"   追踪输入: {channels}通道, {input_size[0]}x{input_size[1]} ...")
    with torch.no_grad():
        flops, params = profile(wrapper, inputs=(dummy,), verbose=False)

    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f" 理论计算量 (FLOPs): {flops_str}")
    print(f"📦 参数量 (Params)    : {params_str} (与上面一致)")

    # 4. 推理速度测试
    print(f"\n{'─' * 60}")
    print("🏎️ 推理速度测试")
    print(f"{'─' * 60}")

    # 预热
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dict(inputs=dummy, data_samples=[DetDataSample()]))
    cuda.synchronize()

    # 正式计时
    start = time.time()
    with torch.no_grad():
        for _ in range(test_iter):
            _ = model(dict(inputs=dummy, data_samples=[DetDataSample()]))
    cuda.synchronize()
    end = time.time()

    total_time_ms = (end - start) * 1000
    avg_latency = total_time_ms / test_iter
    fps = 1000 / avg_latency

    print(f"⏱️  平均延迟: {avg_latency:.2f} ms/张")
    print(f"🚀 推理 FPS : {fps:.2f}")
    print(f"📅 测试规模: {test_iter} 次 (已预热 {warmup} 次)")

    # 5. 显存
    if cuda.is_available():
        alloc = cuda.memory_allocated(device) / 1024 ** 2
        print(f"\n💾 当前显存占用: {alloc:.1f} MB")
        print(f" 训练估算显存: ~{alloc * 4 / 1024:.1f} GB (batch=4 时)")

    print(f"\n{'=' * 60}")
    print("✅ 分析完成")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    # ================= 配置区 =================
    CONFIG_FILE = r"E:\pycharm\project\Swin-Deformable\mmdetection\my_running\configs\Swin-DINO.py"
    CHECKPOINT = None  # 填权重路径可测真实速度，None 则用随机权重
    INPUT_SIZE = (384, 384)
    CHANNELS = 3  # 如果你的 RGBTDataPreprocessor 将红外+可见光拼接为6通道，请改为 6
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # ==========================================

    # 自动寻找最新权重（可选）
    if CHECKPOINT is None:
        work_dir = 'work_dirs/Swin-DINO'
        if os.path.exists(work_dir):
            ckpts = sorted([f for f in os.listdir(work_dir) if f.endswith('.pth')])
            if ckpts:
                CHECKPOINT = os.path.join(work_dir, ckpts[-1])
                print(f"🔍 自动找到权重: {CHECKPOINT}\n")

    analyze_model(CONFIG_FILE, CHECKPOINT, INPUT_SIZE, CHANNELS, device=DEVICE)
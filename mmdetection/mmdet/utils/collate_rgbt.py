import torch

def collate_rgbt(batch):
    """
    将 batch 中的多模态图像打包成 batch 形式。
    batch: list of dict, 每个元素包含
        dict(inputs=(rgb, ir, edge), data_samples=DetDataSample)
    """
    rgb_list, ir_list, edge_list, data_samples = [], [], [], []

    for sample in batch:
        rgb, ir, edge = sample['inputs']
        rgb_list.append(rgb)
        ir_list.append(ir)
        edge_list.append(edge)
        data_samples.append(sample['data_samples'])

    batch_inputs = (
        torch.stack(rgb_list, dim=0),
        torch.stack(ir_list, dim=0),
        torch.stack(edge_list, dim=0)
    )

    return dict(inputs=batch_inputs, data_samples=data_samples)

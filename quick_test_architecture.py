"""
跨模态错误检测系统 - 快速测试脚本
Quick Test Script for Cross-Modal Error Detector

验证模块化架构的核心功能
"""

import torch
import numpy as np
from cross_modal_error_detector import (
    CrossModalErrorDetector,
    StructureAwareTransformer,
    PretrainedTextEncoder,
    SimpleMLPEncoder,
    CrossAttentionFusion,
    SimpleConcatFusion,
    MLPDetectionHead,
    ContrastiveDetectionHead,
    TabularProcessor,
    TextProcessor,
    CorruptionBasedDataset,
    train_step_corruption,
    collate_fn_corruption,
    build_tabular_inputs_from_rows,
)
from torch.utils.data import DataLoader


def quick_test():
    """快速测试所有组件"""
    
    print("\n" + "=" * 80)
    print("跨模态错误检测系统 - 快速测试")
    print("=" * 80)
    
    device = 'cpu'
    print(f"\n使用设备: {device}")
    
    # 1. 生成少量测试数据
    print("\n[1/5] 生成测试数据...")
    clean_rows = [
        [1000, "Employee_0", 25, "Engineering", 75000, "San Francisco"],
        [1001, "Employee_1", 30, "Sales", 65000, "New York"],
        [1002, "Employee_2", 35, "Marketing", 70000, "Boston"],
        [1003, "Employee_3", 28, "HR", 60000, "Seattle"],
    ]
    
    text_descriptions = [
        "Employee table with ID, Name, Age, Department, Salary, City columns"
    ] * 4
    
    print(f"  ✓ 创建了 {len(clean_rows)} 条测试数据")
    
    # 2. 创建数据集
    print("\n[2/5] 创建数据集...")
    dataset = CorruptionBasedDataset(
        clean_rows=clean_rows,
        text_descriptions=text_descriptions,
        corruption_prob=0.3,
        tabular_processor=TabularProcessor(d_cell=32),
        text_processor=TextProcessor(max_seq_len=64)
    )
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_corruption)
    print(f"  ✓ 数据集大小: {len(dataset)}")
    
    # 3. 测试不同的模型配置
    print("\n[3/5] 测试不同的模型配置...")
    
    d_model = 128
    
    configs = [
        {
            'name': 'StructureAware + CrossAttention',
            'tabular_encoder': StructureAwareTransformer(32, d_model, nhead=4, num_layers=1),
            'fusion': CrossAttentionFusion(d_model, nhead=4)
        },
        {
            'name': 'MLP + SimpleConcatFusion',
            'tabular_encoder': SimpleMLPEncoder(32, d_model, num_layers=2),
            'fusion': SimpleConcatFusion(d_model)
        },
    ]
    
    for i, config in enumerate(configs):
        print(f"\n  配置 {i+1}: {config['name']}")
        
        # 构建模型
        model = CrossModalErrorDetector(
            tabular_encoder=config['tabular_encoder'],
            text_encoder=PretrainedTextEncoder(30522, d_model, nhead=4, num_layers=1, max_seq_len=64),
            fusion_module=config['fusion'],
            detection_head=MLPDetectionHead(d_model, hidden_dim=64, output_dim=1)
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    - 参数量: {num_params:,}")
        
        # 测试前向传播
        batch = next(iter(dataloader))
        row_samples, text_inputs, labels = batch
        tabular_inputs = build_tabular_inputs_from_rows(
            row_samples,
            dataset.tabular_processor.to(device),
            dataset.column_names,
        )
        
        with torch.no_grad():
            logits = model(tabular_inputs, text_inputs)
            print(f"    - 输出形状: {logits.shape}")
            print(f"    ✓ 前向传播成功")
    
    # 4. 测试训练步骤
    print("\n[4/5] 测试训练步骤...")
    
    model = CrossModalErrorDetector(
        tabular_encoder=StructureAwareTransformer(32, d_model, nhead=4, num_layers=1),
        text_encoder=PretrainedTextEncoder(30522, d_model, nhead=4, num_layers=1, max_seq_len=64),
        fusion_module=CrossAttentionFusion(d_model, nhead=4),
        detection_head=MLPDetectionHead(d_model, hidden_dim=64, output_dim=1)
    )
    
    optimizer_groups = [{"params": model.parameters()}]
    proc_params = list(dataset.tabular_processor.parameters())
    if proc_params:
        optimizer_groups.append({"params": proc_params})
    optimizer = torch.optim.Adam(optimizer_groups, lr=1e-3)
    
    # 训练几个步骤
    for epoch in range(3):
        losses = []
        for batch in dataloader:
            loss = train_step_corruption(
                model,
                batch,
                optimizer,
                dataset.tabular_processor,
                device,
                column_names=dataset.column_names,
            )
            losses.append(loss)
        avg_loss = np.mean(losses)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print("  ✓ 训练步骤成功")
    
    # 5. 测试推理
    print("\n[5/5] 测试推理...")
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        row_samples, text_inputs, labels = batch
        tabular_inputs = build_tabular_inputs_from_rows(
            row_samples,
            dataset.tabular_processor.to(device),
            dataset.column_names,
        )
        
        logits = model(tabular_inputs, text_inputs).squeeze(-1)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = (predictions == labels).float().mean().item()
        
        print(f"  ✓ 推理准确率: {accuracy:.2%}")
        print(f"  ✓ 预测分布: {predictions.float().mean().item():.2%} 为干净")


if __name__ == "__main__":
    quick_test()



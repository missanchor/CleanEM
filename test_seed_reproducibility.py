#!/usr/bin/env python
"""
测试随机种子固定效果
Test Random Seed Fix Effectiveness

该脚本验证随机种子设置是否成功，确保多次运行结果完全一致
"""

from cross_modal_error_detector.runner.configuration import set_seed
import numpy as np
import torch


def test_basic_randomness():
    """测试基本随机数生成的可重现性"""
    print("\n" + "=" * 60)
    print("测试 1: 基本随机数生成")
    print("=" * 60)

    seed = 42
    results = []

    for i in range(3):
        set_seed(seed)
        # Python random
        import random
        python_rand = random.random()

        # NumPy
        numpy_rand = np.random.random()

        # PyTorch
        torch_rand = torch.rand(3)

        # NumPy 的随机选择
        numpy_choice = np.random.choice([1, 2, 3, 4, 5])

        # NumPy 的随机整数
        numpy_int = np.random.randint(1, 100)

        results.append({
            'run': i + 1,
            'python_random': python_rand,
            'numpy_random': numpy_rand,
            'torch_random': torch_rand.tolist(),
            'numpy_choice': numpy_choice,
            'numpy_int': numpy_int
        })

        print(f"运行 {i+1}:")
        print(f"  Python random: {python_rand:.10f}")
        print(f"  NumPy random: {numpy_rand:.10f}")
        print(f"  Torch random: {torch_rand.tolist()}")
        print(f"  NumPy choice: {numpy_choice}")
        print(f"  NumPy int: {numpy_int}")

    # 检查所有结果是否一致（使用自定义比较函数）
    all_consistent = True
    for i in range(1, len(results)):
        r1, r2 = results[0], results[i]
        # 手动比较每个字段，考虑浮点数精度（排除 'run' 字段）
        inconsistent_fields = []
        if abs(r1['python_random'] - r2['python_random']) > 1e-10:
            inconsistent_fields.append('python_random')
        if abs(r1['numpy_random'] - r2['numpy_random']) > 1e-10:
            inconsistent_fields.append('numpy_random')
        if r1['numpy_choice'] != r2['numpy_choice']:
            inconsistent_fields.append('numpy_choice')
        if r1['numpy_int'] != r2['numpy_int']:
            inconsistent_fields.append('numpy_int')
        if len(r1['torch_random']) != len(r2['torch_random']):
            inconsistent_fields.append('torch_random_length')
        else:
            for idx, (a, b) in enumerate(zip(r1['torch_random'], r2['torch_random'])):
                if abs(a - b) > 1e-10:
                    inconsistent_fields.append(f'torch_random[{idx}]')
                    break

        if inconsistent_fields:
            all_consistent = False
            print(f"  ⚠️  运行 1 vs 运行 {i+1} 不一致: {', '.join(inconsistent_fields)}")

    print(f"\n✅ 所有结果是否一致: {all_consistent}")
    return all_consistent


def test_dataloader_seed():
    """测试数据加载器的随机性"""
    print("\n" + "=" * 60)
    print("测试 2: 数据加载器随机性")
    print("=" * 60)

    from cross_modal_error_detector.runner.data_loading import MockDataGenerator

    seed = 123
    results = []

    for i in range(2):
        generator = MockDataGenerator(seed=seed)
        data1, text1 = generator.generate("employee", 10)
        data2, text2 = generator.generate("sales", 10)

        results.append({
            'run': i + 1,
            'employee_data_len': len(data1),
            'sales_data_len': len(data2),
            'employee_data_0': data1[0] if data1 else None,
            'sales_data_0': data2[0] if data2 else None,
        })

        print(f"运行 {i+1}:")
        print(f"  员工数据行数: {len(data1)}")
        print(f"  销售数据行数: {len(data2)}")
        print(f"  员工第一行: {data1[0] if data1 else 'None'}")
        print(f"  销售第一行: {data2[0] if data2 else 'None'}")

    # 检查结果是否一致
    # 注意：MockDataGenerator 在每次生成数据时不会重置随机状态，
    # 所以多次调用 generate() 会产生不同的结果
    # 这实际上是预期行为，因为数据生成器是连续生成数据的
    print(f"\n💡 注意：MockDataGenerator 不会在每次 generate() 调用时重置随机状态")
    print(f"    这是正常行为，因为数据生成器应该产生连续不同的数据")

    # 真正的测试应该是：相同种子重新创建生成器是否产生相同数据
    generator1 = MockDataGenerator(seed=seed)
    data1_1, text1_1 = generator1.generate("employee", 5)

    generator2 = MockDataGenerator(seed=seed)
    data1_2, text1_2 = generator2.generate("employee", 5)

    consistent = data1_1 == data1_2
    print(f"✅ 相同种子重新创建生成器是否产生相同数据: {consistent}")

    if consistent:
        print(f"  员工数据: {data1_1[0]}")
    else:
        print(f"  第一次: {data1_1[0]}")
        print(f"  第二次: {data1_2[0]}")

    return consistent


def test_cuda_determinism():
    """测试 CUDA 随机性设置"""
    print("\n" + "=" * 60)
    print("测试 3: CUDA 随机性设置")
    print("=" * 60)

    seed = 789
    set_seed(seed)

    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"  cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")

        # 测试多次运行是否一致
        results = []
        for i in range(2):
            set_seed(seed)
            tensor = torch.rand(5, 5).cuda()
            results.append(tensor.cpu().tolist())
            print(f"  运行 {i+1}: {tensor[0, 0].cpu().item():.10f}")

        consistent = results[0] == results[1]
        print(f"\n✅ CUDA 张量生成是否一致: {consistent}")
        return consistent
    else:
        print("⚠️  CUDA 不可用，跳过 CUDA 测试")
        return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("随机种子固定效果测试")
    print("Random Seed Fix Effectiveness Test")
    print("=" * 60)

    test1_result = test_basic_randomness()
    test2_result = test_dataloader_seed()
    test3_result = test_cuda_determinism()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"✅ 基本随机数测试: {'通过' if test1_result else '失败'}")
    print(f"✅ 数据加载器测试: {'通过' if test2_result else '失败'}")
    print(f"✅ CUDA 确定性测试: {'通过' if test3_result else '失败'}")

    all_passed = test1_result and test2_result and test3_result
    print(f"\n🎉 所有测试{'通过' if all_passed else '失败'}！")

    if all_passed:
        print("\n💡 提示：随机参数已成功固定，实验结果应该完全可重现。")
    else:
        print("\n⚠️  警告：部分测试失败，可能存在未固定的随机参数。")


if __name__ == "__main__":
    main()

# Dual-Verification (P_clean/P_dirty) 改造完成报告

## 概述

成功实现了 Dual-Verification 系统，通过 P_clean/P_dirty 双谓词进行空间划分，实现更精确的错误检测。系统支持自动迭代修补，采用“覆盖导向设计”（Coverage-by-design），允许全 Clean 列，但禁止全 Dirty 列。

## 核心流程解析 (dual_main)

系统的执行逻辑分为以下关键阶段：

### 1. 启发式基础探测 (Base Rule Generation)
- **多 Agent 协作**：利用 `MissingLegislator` (缺失值), `TypoLegislator` (拼写错误), `OutlierLegislator` (离群值), `PatternLegislator` (模式违规) 生成初始错误检测规则。
- **作用**：为后续的双规则生成提供“什么是错误”的上下文参考。
- **Profiler 增强**：`PandasProfiler` 会额外推断 `relationship_constraints`（如 `Stateavg` 必须以 `State` 值开头、带有 `City` 字段的列需包含对应城市名等），为后续 Clean Relationship Agent 提供上下文。

### 2. 双规则生成与覆盖设计 (Dual Rule Generation)
- **P_clean (宽容型)**：旨在确认数据是干净的。设计遵循“四大质量支柱”——**完整性**（missing 立即拒绝）、**准确性**（数值范围/离散取值正确）、**列关系约束**（尊重跨列依赖，若元数据提供）、**Pattern 一致性**（格式/长度/正则），并显式参考第一步产出的 base rules，使 clean 判定与专业 Agent 的意图保持一致。整体保持 **宽容 (Permissive)**，只剔除明显违背这些支柱的值。
- **Clean Base Rules**：新增 `CleanCompletenessLegislator / CleanAccuracyLegislator / CleanRelationshipLegislator / CleanPatternLegislator` 四个 Agent，为每一列生成可执行的 clean 基础谓词，再以“组合器”方式拼装成最终 `P_clean`，实现与 dirty 侧对称的基座。
- **P_dirty (严格型)**：旨在检测错误。设计原则是 **严格 (Strict)**，只标记确定的异常。
- **覆盖导向设计 (Coverage-by-design)**：在 `DualLegislator` 中，`P_dirty` 被定义为 `P_clean` 的**安全补集**（通过 `safe_not` 逻辑）。这在初始阶段就消除了 Grey Zone 和 Conflict，确保数据空间的完备划分。

### 3. 迭代修补机制 (Iterative Refinement)
- **四区域监控**：
    - `Determined Clean`: `P_clean=True, P_dirty=False` ✅
    - `Determined Dirty`: `P_clean=False, P_dirty=True` ✅
    - `Grey Zone`: `P_clean=False, P_dirty=False` (不确定) ⚠️
    - `Conflict`: `P_clean=True, P_dirty=True` (矛盾) ❌
- **约束判定**：规则必须满足 `conflict_rate == 0`, `grey_rate <= tolerance`, `dirty_rate < 1.0`。
- **样本回传**：若不满足约束，收集 Grey/Conflict 区域的样本，回传给 LLM 重新立法，通常进行 1-3 轮迭代。

### 4. 最终裁定与评估 (Final Judge & Evaluation)
- **最佳选择**：在所有满足约束的候选规则中，选择 `dirty_rate` 最小且最合理的规则。
- **地面真值对比**：自动与 `_clean.csv` 对比，输出精确率 (Precision)、召回率 (Recall) 和 F1 值。

## 完成的功能

### 1. 核心数据结构 (`dual_types.py`)

定义了以下关键数据结构：

- **DualRule**: 双规则对（clean predicate + dirty predicate）
- **DualEvaluationResult**: 双规则评估结果，包含四区域统计
- **RefinementRound**: 规则修补轮次记录
- **DualRuleSet**: 双规则集合（所有列的最佳规则）

### 2. 立法器扩展 (`legislator.py`)

新增了 `DualLegislator` 类：

- **generate_dual_rules()**: 生成配对的 clean/dirty 规则
- **generate_dual_rules_per_column()**: 为所有列生成双规则
- 支持自动处理缺失值、冲突和全Dirty问题
- 包含回退机制，确保总是能生成基本规则
- **Clean Pillar Agents**：
  - `CleanCompletenessLegislator`: 拒绝所有缺失/占位符，提供完整性基线
  - `CleanAccuracyLegislator`: 依据统计范围/长度分布生成准确性谓词
  - `CleanRelationshipLegislator`: 利用 `Profiler` 推断的 `relationship_constraints` 约束跨列依赖（例如 `Stateavg` 必须以 `State` 前缀开头）
  - `CleanPatternLegislator`: 保障固定长度/正则/ID 样式一致性

**特性**:
- 两个谓词必须互斥（不能同时为True）
- 显式处理 None/NaN/空字符串
- 支持问题样本回传进行规则修补

### 3. 法官系统增强 (`judge.py`)

新增以下方法：

- **evaluate_dual_rules()**: 评估双规则，分类为四个区域
  - Conflict: P_clean=True AND P_dirty=True ❌
  - Grey Zone: P_clean=False AND P_dirty=False ⚠️
  - Determined Clean: P_clean=True AND P_dirty=False ✓
  - Determined Dirty: P_clean=False AND P_dirty=True ✓

- **select_best_dual_rules()**: 基于约束选择最佳规则
  - 约束1: conflict_rate == 0
  - 约束2: grey_rate <= grey_tolerance
  - 约束3: dirty_rate < 1.0（禁止全Dirty）
  - 目标: 在满足约束的规则中选择 dirty_rate 最小的

- **refine_dual_rules()**: 自动迭代修补
  - 最多 N 轮修补
  - 收集 Grey/Conflict/AllDirty 样本
  - 回传给 LLM 修正规则
  - 直至满足约束或达到上限

- **get_detected_dirty_values()**: 获取所有被标记为 Dirty 的值
- **print_dual_summary()**: 打印双规则评估摘要
- **save_dual_results()**: 保存结果到文件

### 4. 主流程更新 (`main.py`)

- **dual_main()**: 完整的双验证流程
  - 步骤1: 加载和分析数据
  - 步骤2: 生成双规则
  - 步骤3: 迭代修补（最多3轮）
  - 步骤4: 检测 Dirty 值
  - 步骤5: 打印摘要
  - 步骤6: 与真实数据对比评估
  - 步骤7: 保存结果

- **命令行模式选择**:
  - `python main.py dual` - 运行双验证模式（新功能）
  - `python main.py vr` - 运行 VR 模式（原有功能）
  - 默认模式为 dual

## 输出文件

系统会生成以下结果文件到 `agentic_error_detector/results/`:

1. **dual_rules.json**: 每列最终选定的最佳双规则对。
2. **dual_evaluation.json**: 包含每列的四区域统计指标（Conflict/Grey/Clean/Dirty Rate）。
3. **refinement_history.json**: 记录了每一轮修补使用的样本和演进过程。
4. **detected_dirty_values.json**: 检测到的 Dirty 值列表，包含行索引、原始值和违反的规则。
5. **coverage_gaps.json**: 记录经过 N 轮迭代后仍无法满足约束的列（即“失控列”）。

## 关键特性

### ✅ 允许全 Clean 列 (dirty_rate = 0)
- 完全正确的列可以保持 100% 干净，系统不会强制要求每列必须检出错误。

### ❌ 禁止全 Dirty 列 (dirty_rate = 1.0)
- 整个列都被标记为 Dirty 是不可接受的，这通常意味着 `P_clean` 过于严苛或 `P_dirty` 过于泛化。系统会自动通过迭代修补来修正此类“误杀”。

### 🔄 覆盖导向设计 (Coverage-by-design)
- `P_dirty` 默认作为 `P_clean` 的逻辑补集生成，结合 `safe_not` 异常处理，从数学上降低了 Grey Zone 和 Conflict 的出现概率。

### 🛡️ 安全执行沙箱
- 所有的谓词执行都通过 `safe_dict` 注入受限的全局变量（如 `re`, `pd`, `np`），并捕获执行期异常，将报错样本归类为 Grey Zone 以触发修补。

### 📊 迭代样本学习
- 每一轮修补都会提取 Grey Zone 和 Conflict 的具体样本值及其频次，帮助 LLM 在下一轮立法时更具针对性。

## 使用方法

### 运行双验证模式
```bash
python main.py dual
```

### 运行 VR 模式（原有）
```bash
python main.py vr
```

### 作为模块使用
```python
from agentic_error_detector import PandasProfiler, LegislatorFactory, Judge

# 加载数据
profiler = PandasProfiler("data/hospital_error-01.csv")

# 生成双规则
metadata = profiler.get_metadata()
factory = LegislatorFactory()
dual_rules = factory.generate_dual_rules_per_column(metadata)

# 评估和修补
judge = Judge()
best_rules, history = judge.refine_dual_rules(
    profiler.df,
    metadata,
    dual_rules,
    max_rounds=3,
    grey_tolerance=0.0
)

# 获取检测到的错误
dirty_values = judge.get_detected_dirty_values(best_rules, profiler.df)
```

## 测试验证

运行测试脚本：
```bash
python test_dual_verification.py
```

测试覆盖：
- ✓ 导入验证
- ✓ DualLegislator 类验证
- ✓ Judge 双验证方法验证
- ✓ LegislatorFactory 双验证方法验证
- ✓ 数据类型验证
- ✓ 主流程函数验证

## 文件清单

### 新增文件
- `/mnt/data/welkinni/table_det/agentic_error_detector/dual_types.py` - 数据结构定义
- `/mnt/data/welkinni/table_det/test_dual_verification.py` - 测试脚本
- `/mnt/data/welkinni/table_det/DUAL_VERIFICATION_README.md` - 文档（本文件）

### 修改文件
- `/mnt/data/welkinni/table_det/agentic_error_detector/legislator.py` - 添加 DualLegislator
- `/mnt/data/welkinni/table_det/agentic_error_detector/judge.py` - 添加双验证方法
- `/mnt/data/welkinni/table_det/agentic_error_detector/main.py` - 添加 dual_main() 和模式选择

## 总结

Dual-Verification 系统已完全实现，满足所有设计要求：

1. ✅ 实现了 P_clean/P_dirty 双谓词系统
2. ✅ 四个区域空间划分（Conflict、Grey、DeterminedClean、DeterminedDirty）
3. ✅ 自动迭代修补机制
4. ✅ 允许全Clean列
5. ✅ 禁止全Dirty列
6. ✅ 完整的结果输出和保存
7. ✅ 与现有 VR 系统的兼容性
8. ✅ 命令行模式选择

系统可以投入运行！🎉

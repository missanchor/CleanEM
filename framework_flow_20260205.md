# `main.py` 框架与流程超详细总结（不含日志逻辑）

本文按“从外到内、从流程到细节”的顺序，总结 `main.py` 中实现的 Agentic Error Detector 主流程，仅关注**数据与规则处理逻辑**，不涉及日志记录与输出重定向相关实现。

- 入口文件：`main.py`
- 主要职责：提供命令行入口，驱动**双重验证（dual verification）+ 规则精修（clean rule refinement）**的整套数据错误检测流程
- 基本对象：
  - `PandasProfiler`：负责读入脏数据表、生成列级元信息 `metadata`
  - `AgentFactory`：基于 LLM 为每一列生成“脏规则（dirty rules）”和“干净规则（clean rules）”
  - `Judge`：负责规则评估、接受/拒绝规则、组合规则、生成错误单元格集合、以及与真值表比对
  - `DisjointnessValidator`：验证最终双重规则在取值空间上的“非交叠性”和灰区情况

---

## 一、命令行入口与参数解析

### 1. `main()` 顶层入口

```python
def main() -> None:
    args = parse_args()
    run_dual_mode(args)
```

- `main()` 做两件事：
  - 调用 `parse_args()` 解析命令行参数
  - 调用 `run_dual_mode(args)`，根据解析出的参数执行完整 dual 模式流程

### 2. `parse_args()` 参数设计

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Error Detector CLI (dual verification by default)."
    )
    ...
    return parser.parse_args()
```

核心参数（仅列出与算法/流程直接相关的部分）：

- `--dirty_csv`
  - 默认：`data/hospital_error-01.csv`
  - 含义：需要检测错误的“脏”数据表路径。整个流程围绕该表展开。
- `--clean_csv`
  - 默认：`data/hospital_clean.csv`
  - 含义：可选的“干净”真值表，用于**评估**检测效果，而非参与规则生成。
- `--output_dir`
  - 默认：`results/agentic_error_detector`
  - 含义：存放各类序列化结果（例如评估结果、规则、运行记录等）的目录。
- `--max_rounds`
  - 默认：`10`
  - 含义：后续“clean 规则精修”阶段中，每列规则的最大迭代轮数。
- `--grey_tolerance`
  - 默认：`0.01`
  - 含义：最终双重规则评估时，允许的“灰区”比例上限。
- `--vr_threshold`
  - 默认：`0.6`
  - 含义：Judge 在遗留 VR 模式（Violation Rate）下使用的违例率阈值，同时被用作规则评估的阈值。
- `--skip_initial_clean` / `--skip_initial_dirty`
  - 目前在本文件内未被直接使用，为后续控制“是否跳过初始规则生成”的扩展钩子。
- `--base_url`
  - 默认：`http://localhost:8000/v1`
  - 含义：LLM 接口的 OpenAI-compatible base URL，`AgentFactory` 通过该地址调用模型。
- `--model`
  - 默认：`None`
  - 含义：可选的模型名称，用于覆盖 `AgentFactory` 默认模型设定。
- `--max_workers`
  - 默认：`8`
  - 含义：进行列级并行调用 LLM 时的最大并发 worker 数。

解析完成后，得到的 `args` 会贯穿整个 pipeline。

---

## 二、整体 Dual 模式流程：`run_dual_mode(args)`

`run_dual_mode` 是本文件的主流程函数，负责从数据加载、规则生成、规则评估，到规则精修、双重验证和最终评估的所有步骤。

代码结构中，流程被显式标记为 `[1/7]` 至 `[7/7]` 七个阶段，便于跟踪。

### 阶段 [1/7]：Profiling 脏数据集

```python
profiler = PandasProfiler(args.dirty_csv)
metadata = profiler.get_metadata()
```

- 输入：`args.dirty_csv` 指向的 CSV 文件路径
- 过程：
  - `PandasProfiler` 读入 CSV，构建 `profiler.df`（脏数据的 DataFrame）
  - `get_metadata()` 生成列级元数据 `metadata`，典型信息包括：
    - 每列的数据类型
    - 缺失率
    - 值分布统计
    - 其他用于提示 LLM 的上下文信息
- 输出：
  - `profiler.df`：后续所有规则评估与错误检测的基础数据
  - `metadata`：驱动 LLM 生成规则的关键上下文

### 阶段 [2/7]：生成基础脏规则（Base Agent Rules）

```python
factory = AgentFactory(base_url=args.base_url, model=args.model, max_workers=args.max_workers)

base_rules, dirty_prompts = factory.generate_rules_per_column(metadata)
```

- `AgentFactory` 负责：
  - 封装 LLM 调用细节（包含 base_url、model、并发度 `max_workers` 等配置）
  - 根据 `metadata` 为每一列生成一组“脏规则”
- 输出：
  - `base_rules`：
    - 结构：`Dict[column] -> List[(agent, rule_string)]`
    - 解释：对每一列，生成若干 agent 输出的规则，每条规则使用 `rule_string` 表示可执行的 Python 表达式或函数。
  - `dirty_prompts`：
    - 保存 LLM 请求与响应的原始内容，便于后续分析或重放。

这些“脏规则”侧重于检测**明显异常/错误模式**，例如：
- 缺失值检测
- 明显拼写错误
- 异常模式/格式
- 数值离群点等

### 阶段 [3/7]：生成 clean 规则（Completeness / Accuracy / Pattern / Relationship）

```python
clean_rules, clean_prompts = factory.generate_clean_rules_per_column(metadata)
```

- 仍由同一个 `AgentFactory` 使用 `metadata` 生成，但目标是**刻画“正常/干净”数据的约束**，而不是找异常。
- 输出：
  - `clean_rules`：
    - 结构：`Dict[column] -> List[(agent, rule_string)]`
    - 含义：每一条规则描述“什么样的取值才是合理/干净”的逻辑
  - `clean_prompts`：
    - 与 `dirty_prompts` 类似，记录 LLM 调用过程。

这一阶段产出的规则在后续会参与：
- 独立评估（单看 clean 规则检测能力）
- 与脏规则结合，形成双重规则（dual rules）
- clean 规则精修（refinement）阶段

### 阶段 [4/7]：Standalone 规则评估与初步错误检测

```python
judge = Judge(threshold=args.vr_threshold, violation_threshold=args.vr_threshold)

base_evaluation_results = judge.evaluate_rules(profiler.df, base_rules)
accepted_base_rules = judge.get_accepted_rules(base_evaluation_results)

clean_evaluation_results = judge.evaluate_rules(profiler.df, clean_rules, rule_type="clean")
accepted_clean_rules = judge.get_accepted_rules(clean_evaluation_results)
```

这一阶段使用 `Judge` 对“脏规则”和“clean 规则”分别做独立评估。

- `evaluate_rules(df, rules, rule_type=...)`：
  - 对每一列的每一条规则，计算违例情况（例如违例率、支持度等）
  - 结合 `threshold`/`violation_threshold` 策略判断规则质量
- `get_accepted_rules(evaluation_results)`：
  - 从评估结果中过滤出“被接受的规则”，形成：
    - `accepted_base_rules`
    - `accepted_clean_rules`

随后进行一次**组合 AND/OR 错误检测**：

```python
base_detected_errors = judge.get_detected_errors(
    accepted_base_rules,      # dirty rules (OR logic)
    accepted_clean_rules      # clean rules (AND logic)
)
```

组合逻辑说明：

- Dirty Rule（OR）：违反任一脏规则，即被视作“潜在脏值”
- Clean Rule（AND）：必须满足所有 clean 规则才算“完全干净”
- 最终错误判定：

> Error = (NOT all clean rules satisfied) AND (at least one dirty rule violated)

输出：

- `base_detected_errors`：基于当前接受规则的初始错误单元格集合。

### 阶段：可选的真值表评估（base 规则）

如果提供了 `--clean_csv`，则加载干净表并评估基础检测效果：

```python
if args.clean_csv:
    clean_df = pd.read_csv(args.clean_csv)
    base_metrics_summary = judge.evaluate_with_ground_truth(
        profiler.df,
        clean_df,
        base_detected_errors
    )
    judge.print_evaluation_summary(base_metrics_summary)
```

- 输入：
  - `profiler.df`：脏数据
  - `clean_df`：真值干净数据
  - `base_detected_errors`：当前检测到的错误集合
- 作用：
  - 产出关于召回率/精确率/F1 等指标的汇总（具体内容由 `Judge` 实现）
  - 用于衡量“单靠初始规则”就能达到什么检测质量

### 阶段 [5/7]：Clean 规则级精修（核心：`run_clean_rule_refinement`）

在完成初步评估后，进入 clean 规则精修阶段：

```python
best_rules, refinement_history = run_clean_rule_refinement(
    profiler.df,
    metadata,
    {
        column: [(r['agent'], r['rule_string']) for r in accepted_base_rules.get(column, [])]
        for column in accepted_base_rules.keys()
    },
    {
        column: [(r['agent'], r['rule_string']) for r in accepted_clean_rules.get(column, [])]
        for column in accepted_clean_rules.keys()
    },
    factory,
    judge,
    args,
    clean_prompts,
    dirty_prompts,
    clean_df
)
```

准备输入时做了一个重要转换：

- 将 `accepted_base_rules` / `accepted_clean_rules` 的内部结构统一映射为：

```python
Dict[column] -> List[(name, rule_string)]
```

- 这样 `run_clean_rule_refinement` 可以通过统一接口消费这两类规则。

精修后的输出：

- `best_rules`：
  - 结构：`Dict[column] -> DualRule`
  - 每列最终得到一个**双重规则对象**，内部包含干净规则和脏规则两部分，用于后续统一评估。
- `refinement_history`：
  - 记录每一列在精修过程中所有中间状态与迭代信息，便于分析调整过程。

如 `best_rules` 为空，则说明所有列的规则精修都未生成可接受的双规则，此时流程提前结束。

### 阶段 [6/7]：双规则的非交叠性验证

```python
validator = DisjointnessValidator(gap_tolerance=args.grey_tolerance)
validation_result = validator.validate_batch(profiler.df, best_rules)
print(validator.report_violations(validation_result))
```

- `DisjointnessValidator` 的目标：
  - 确认 refined 双规则在数值空间上“一致且分明”
  - 重点检查：
    - 同一单元格是否可能被同时判为“干净”和“脏”（冲突）
    - 是否存在过大的灰区（既不明显干净也不明显脏）
- `gap_tolerance`：
  - 控制可接受灰区比例上限
  - 大于阈值的灰区/冲突会被视为需要关注的问题

验证结果用于研究/调试，不会直接阻断后续评估，但会提供详细报告。

### 阶段 [7/7]：最终双规则评估与错误检测

首先，将 `best_rules` 适配为 `Judge` 的统一输入格式：

```python
evaluation_payload = _materialize_rule_payload(best_rules)
```

`_materialize_rule_payload` 的核心逻辑：

```python
def _materialize_rule_payload(best_rules) -> Dict[str, List[tuple]]:
    payload: Dict[str, List[tuple]] = {}
    for column, rule in best_rules.items():
        payload[column] = [(rule.agent_name, rule.clean_rule_str, rule.dirty_rule_str)]
    return payload
```

- 输入：`best_rules`（每列一个 `DualRule` 对象）
- 输出：`payload`（每列变为 `[(agent_name, clean_rule_str, dirty_rule_str)]` 的列表）
- 目的：给 `Judge.evaluate_dual_rules` 提供统一的结构化输入。

然后进行最终评估与错误检测：

```python
evaluation_results = judge.evaluate_dual_rules(
    profiler.df,
    evaluation_payload,
    grey_tolerance=args.grey_tolerance,
    metadata=metadata
)
detected_dirty_values = judge.get_detected_dirty_values(best_rules, profiler.df)
judge.print_dual_summary(best_rules, evaluation_results)
```

- `evaluate_dual_rules`：
  - 按列、按双规则，对 `profiler.df` 逐单元格评估
  - 结合 `grey_tolerance` 判定干净/脏/灰区
  - 输出结构化的评估结果 `evaluation_results`
- `get_detected_dirty_values`：
  - 从 `best_rules` 的决策结果中抽取最终判定为“脏”的单元格集合。
- `print_dual_summary`：
  - 汇总并展示双规则在整体上的表现（如各列规则数量、检测效果等）。

如果提供了 `clean_df`，还会对 refined 双规则做一次真值评估：

```python
if args.clean_csv and clean_df is not None:
    refined_metrics_summary = judge.evaluate_with_ground_truth(
        profiler.df,
        clean_df,
        detected_dirty_values
    )
    judge.print_evaluation_summary(refined_metrics_summary)
```

至此，Dual Verification 全流程完成。

---

## 三、Clean 规则精修内部流程：`run_clean_rule_refinement(...)`

`run_clean_rule_refinement` 是整个框架中最关键的“第二阶段优化”逻辑，用于在已有规则基础上，通过循环与评估不断改善 clean 规则，并与脏规则组合成更优的 dual rules。

函数签名简化后为：

```python
def run_clean_rule_refinement(
    df,
    metadata,
    base_rules,
    clean_rules,
    factory,
    judge,
    args,
    clean_prompts=None,
    dirty_prompts=None,
    clean_df=None,
):
    ...
    return best_rules, all_history
```

### 1. 输入数据与规则的结构

- `df`：脏数据 DataFrame（`profiler.df`）
- `metadata`：列级元数据
- `base_rules`：
  - `Dict[column] -> List[(agent_name, rule_str)]`
  - 来源：`accepted_base_rules` 中被接受的脏规则
- `clean_rules`：
  - `Dict[column] -> List[(rule_name, rule_str)]`
  - 来源：`accepted_clean_rules` 中被接受的 clean 规则
- `factory`：`AgentFactory`，用于在精修过程中继续调用 LLM
- `judge`：`Judge`，用于评估规则改动是否带来改进
- `args`：CLI 参数（主要使用 `max_rounds` 等字段）
- `clean_prompts` / `dirty_prompts`：
  - 记录初始规则生成阶段的 LLM 对话，可在精修中复用上下文
- `clean_df`：
  - 如存在，则可用于在精修期间对规则质量做更客观的评估。

### 2. 将字符串规则编译为可执行 CleanRule

对每一列，分别处理 clean 与 dirty 规则：

```python
for column in metadata.keys():
    clean_rules_dict = {}
    for rule_name, rule_str in clean_rules.get(column, []):
        rule_func = eval(rule_str, safe_dict)
        clean_rules_dict[rule_name] = CleanRule(
            name=rule_name,
            rule_str=rule_str,
            rule_func=rule_func
        )

    dirty_rules = {}
    for agent_name, rule_str in base_rules.get(column, []):
        rule_func = eval(rule_str, safe_dict)
        dirty_rules[agent_name] = CleanRule(
            name=agent_name,
            rule_str=rule_str,
            rule_func=rule_func
        )
```

要点：

- 使用 `eval(rule_str, safe_dict)` 将字符串形式的规则编译为函数 `rule_func`
  - `safe_dict` 提供受限的执行环境，降低任意代码执行风险
- 用 `CleanRule` 数据结构统一表达 clean/dirty 两类规则：
  - `name`：规则名称或 agent 名称
  - `rule_str`：原始字符串表达
  - `rule_func`：可直接运行在 DataFrame 上的函数

如果某列在 clean 与 dirty 两侧都完全没有规则，直接跳过该列。

### 3. 构建列级 CleanRuleSet 与初始 DualRule

对每一列，构建 `CleanRuleSet`：

```python
rule_set = CleanRuleSet(
    column=column,
    clean_rules=clean_rules_dict,
    dirty_rules=dirty_rules
)
clean_rule_sets[column] = rule_set
```

随后，为每列生成一个初始的 `DualRule`：

```python
for column, rule_set in clean_rule_sets.items():
    dual_rule = rule_set.to_dual_rule(agent_factory=factory)
    initial_rules[column] = dual_rule
```

- `CleanRuleSet` 负责在列级别上打包 clean 与 dirty 规则，提供便捷的转换方法 `to_dual_rule`
- `DualRule` 内部通常包含：
  - 规则的字符串表达（`clean_rule_str` / `dirty_rule_str`）
  - 规则执行函数
  - 规则关联的 agent 名称等元信息

如提供了 `clean_df`，函数还会对这些初始 `DualRule` 做一次性能评估：

```python
if clean_df is not None and initial_rules:
    initial_detected_errors = judge.get_detected_dirty_values(initial_rules, df)
    initial_metrics_summary = judge.evaluate_with_ground_truth(
        df,
        clean_df,
        initial_detected_errors
    )
    judge.print_evaluation_summary(initial_metrics_summary)
```

这一步的作用：

- 作为“精修前 baseline”，帮助对比精修前后效果的变化。

### 4. 列级并行精修：`judge.refine_clean_rules(...)`

核心精修逻辑通过一个内部函数 `_refine_single_column` 实现，并通过线程池在所有列上并行执行。

```python
def _refine_single_column(column: str, rule_set: Any):
    console, log_lines = _make_console_buffer()
    col_metadata = metadata.get(column, {})
    refined_set, history = judge.refine_clean_rules(
        df,
        column,
        rule_set,
        factory=factory,
        max_rounds=args.max_rounds,
        conflict_tolerance=0.01,
        metadata=col_metadata,
        output_dir=args.output_dir,
        logger=logger,
        console=console,
    )
    dual_rule = refined_set.to_dual_rule(agent_factory=factory)
    return column, dual_rule, history, log_lines
```

要点：

- `refine_clean_rules` 是精修算法的真正实现者，`run_clean_rule_refinement` 负责为其提供上下游 glue：
  - 输入：
    - 当前列的数据与规则集合 `rule_set`
    - 该列的元数据 `col_metadata`
    - LLM 工厂 `factory`
    - 最大轮数 `max_rounds`
    - 冲突容忍度 `conflict_tolerance=0.01`
  - 输出：
    - `refined_set`：精修后的 `CleanRuleSet`
    - `history`：精修过程中产生的迭代记录
- 将 `refined_set` 再次转为 `DualRule`，保证与后续流程接口一致。

并行执行部分：

```python
max_workers = max(1, int(getattr(args, "max_workers", 1) or 1))
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_column = {
        executor.submit(_refine_single_column, column, rule_set): column
        for column, rule_set in clean_rule_sets.items()
    }
    for future in as_completed(future_to_column):
        column, dual_rule, history, log_lines = future.result()
        best_rules[column] = dual_rule
        all_history[column] = history
```

- 每一列的精修在单独的线程中执行，充分利用并行能力
- `best_rules` 聚合所有列的最终 `DualRule`
- `all_history` 记录每列的精修轨迹

### 5. 返回值

函数最终返回：

```python
return best_rules, all_history
```

- `best_rules`：后续双规则评估与错误检测的直接输入
- `all_history`：主要用于分析/可视化精修过程（例如：规则如何被修改、删减、合并）

---

## 四、总结：`main.py` 中的核心框架抽象

从架构角度，可以将 `main.py` 概括为三层：

1. **CLI 层（配置获取）**
   - `parse_args()` 负责将命令行参数解析为结构化配置 `args`
   - `main()` 作为统一入口，将控制权交给 `run_dual_mode(args)`

2. **Pipeline 编排层**
   - `run_dual_mode(args)` 将整个错误检测流程拆解为多个阶段：
     - 数据加载与 profiling
     - 通过 LLM 生成脏规则与 clean 规则
     - Judge 对规则进行评估与筛选
     - 初步错误检测与可选的真值评估
     - clean 规则精修（调用 `run_clean_rule_refinement`）
     - 双规则的非交叠性验证
     - 最终双规则评估与错误输出
   - 该层不关心规则内部如何被生成/优化，只负责“按顺序调用并串起各个组件”。

3. **规则精修子流程层**
   - `run_clean_rule_refinement(...)` 聚焦于：
     - 将字符串规则转成可执行对象（`CleanRule`）
     - 构建列级规则集合（`CleanRuleSet`）与初始双规则（`DualRule`）
     - 通过 `Judge.refine_clean_rules` 在列级别迭代改进 clean 规则
     - 并行执行所有列的精修任务
   - 最终产出全局 `best_rules`，作为 refined dual rules 用于后续评估。

整体上，`main.py` 不是具体算法实现的地方，而是：

- 聚合 profiler、LLM agent factory、judge、validator 等组件
- 用一个清晰的分阶段 pipeline 编排整个“从数据到错误”的工作流
- 通过 dual verification 思路，将“错误视角（dirty rules）”和“干净视角（clean rules）”统一在同一个决策框架中，并在此基础上进行规则级的精修与验证。


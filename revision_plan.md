下面按你的 4 点约束，基于当前代码（尤其是 agent.py 里的 clean agents、 PandasProfiler 等），给一套 不改现有代码、但可直接落地的新框架设计方案 。我会刻意和现有实现对齐，让你之后实现时改动集中且可控。

我默认的取舍是：

- 立即只考虑 单列 的 missing / outlier / pattern 错误；
- 你提到的 CleanColRelationAgent 先在设计里预留接口，但 当前迭代不实现 （对应你“暂时不考虑 cross-column 错误类型和 Agent”）。
## 1. 整体新框架概览（抛弃 conflict/gap resolve）
新框架不再使用当前的：

- CleanRuleSet + DualRule （P_clean / P_dirty）
- Judge.refine_clean_rules / GapResolver / ConflictResolver
而是换成一条新的 pipeline：

1. Profiling 层 ：仍然使用 PandasProfiler 做列画像，产出 df + metadata 。
2. Clean-Rule Agent 层（四类 clean agent）
   - CleanMissingAgent
   - CleanOutlierAgent
   - CleanPatternAgent
   - CleanColRelationAgent （预留）
     所有规则统一为：
      lambda value, row=None: True 表示“数据正确/正常”，False 表示“不符合该 clean 规则”
3. Dirty-Data Generator Agent 层
   - 一个专门的 LLM agent，基于 metadata + clean seed，生成 现实感强 的脏数据示例；
   - 只输出“数据点 + 错误类型标签”，不输出规则。
4. Calib & EM 层（统一评分）
   - 用 clean agents 的输出（True/False） + 脏数据生成的伪标注，
   - 做 per-rule 的可靠性估计 + EM 弱监督，
   - 输出每个 cell 的脏概率 score S(i, j) ∈ [0, 1] 。
5. Detection & Export 层
   - 最终对每个 cell 输出：
     - S_missing, S_outlier, S_pattern 三个子分数；
     - 汇总的总分 S_total ；
   - 可选：根据 score 选 top-k 疑似错误 cell 导出。
整个设计是“ 纯 clean 规则 + 脏例生成 + EM 校准 ”，不再有 clean/dirty 规则对偶、也不再追求 P_clean/P_dirty 的严格不交叠。

## 2. Clean-Rule Agents：四类 Agent 的角色和接口
### 2.1 Agent 总接口（概念）
定义一个抽象接口（逻辑层面）：

```
class BaseCleanAgent:
    clean_agent_name: str      # "missing", "outlier", "pattern", "relation"

    def generate_rules(self, column: str, metadata: Dict[str, Any]) -> List[str]:
        """
        返回若干个 lambda 字符串：
        lambda value, row=None: <bool expression>
        True = value 符合“正常/合理/正常模式”，False = 违反本 Agent 的 clean 判定。
        """
```
这一点和当前 CleanCompletenessAgent / CleanAccuracyAgent / CleanPatternAgent 是完全兼容的——它们已经输出 lambda value, row=None: ... ，True 表示“正常/合理”。

### 2.2 具体四类 Agent 对应关系
结合当前实现：

- CleanMissingAgent
  
  - 可以直接用现在的 CleanCompletenessAgent 的思想改名/收敛：
    
    - 规则形式类似：
      
      ```
      lambda value, row=None: (
          value is not None and
          str(value).strip().lower() not in MISSING_TOKENS
      )
      ```
  - 它专注于“值是否非缺失”，不负责范围/模式。
- CleanOutlierAgent
  
  - 对 numeric / categorical 列使用现在 CleanAccuracyAgent 的一部分思想，但更明确地作为“outlier clean agent”：
    - numeric：基于 min/max/mean/std、IQR 等给出“合理范围”的 clean 规则；
    - category：可以直接产出“正常值集合/模式”的规则（类似 pattern），或用频率信息构造“明显不合理的类别”的补集。
  - 所有规则返回 True = 在合理范围内；False = outlier/不合理。
- CleanPatternAgent
  
  - 可以直接借用当前 CleanPatternAgent 的逻辑：
    - 通过 PatternExplorer 或 LLM 归纳“正常正则模式”；
    - clean 规则：
      
      ```
      lambda value, row=None: bool(re.match(combined_regex, str(value).strip()))
      ```
- CleanColRelationAgent （预留）
  
  - 对应未来要做的 cross-column consistency（函数依赖、不等式等）；
  - 目前设计接口，但实现可以先留空，或者在 pipeline 配置里直接 disable。
### 2.3 Clean 规则的组织形式
对每列 col ，在新框架中不再构造 CleanRuleSet / DualRule ，而是一个简单的规则池：

```
CleanRulePool[col] = [
  {
    "agent": "CleanMissingAgent",
    "family": "missing",
    "rule_name": f"{agent_name}_{k}",
    "rule_str": "...",
    "rule_func": compiled_lambda,
  },
  {
    "agent": "CleanOutlierAgent",
    "family": "outlier",
    ...
  },
  ...
]
```
- family 用于归类（missing / outlier / pattern / relation），方便后面评分拆解。
- 不再有 “clean vs dirty” 的字段，只有“clean 规则”，脏就是 “违反 clean”。
## 3. 脏数据生成 Agent：DirtyExampleAgent
第二点你希望“脏数据生成由专门的 agent 负责，由 LLM 来生成符合现实的范例”。这一层可以完全独立于 clean rules 的生成。

### 3.1 DirtyExampleAgent 的接口设计
对每列 col ：

```
class DirtyExampleAgent(BaseAgent):
    def generate_dirty_examples(
        self,
        column: str,
        metadata: Dict[str, Any],
        clean_seeds: List[Any],
        error_family: str,  # "missing"|"outlier"|"pattern"
    ) -> List[Dict]:
        """
        返回若干“脏样本”：
        [
          {"value": <corrupted_value>, "label": 1, "error_family": error_family,
           "reason": "unit mismatch", "source_seed": seed_val},
          ...
        ]
        """
```
- 输入：
  - metadata ：包含列类型、top values、统计信息；
  - clean_seeds ：从真实数据中抽出的“看起来很正常”的值（Clean Seed）；
  - error_family ：告诉 LLM 需要造哪一类错误（missing/outlier/pattern）。
- 输出：
  - 一批“值 + 错误标签 + 没必要很精确但合理的 reason”。
### 3.2 Prompt 设计思路（不写具体 prompt，只定逻辑）
- 对 missing：
  - “给定一列中的一些正常取值，请生成真实世界中常见的‘缺失或缺失编码’写法；包括 None, NaN, 空字符串，和领域内常用编码（如 9999, 'N/A'）。”
- 对 outlier：
  - “给定一列数值统计信息（min/max/quantiles）、一些正常值，请生成常见的 outlier 例子（数值符号错误、数量级错误、单位错误、极端值等）。”
- 对 pattern：
  - “给定一些正常值及其 pattern 分析（长度分布、正则形状），请生成常见的 pattern 错误（少/多字符、分隔符错误、局部乱序等）。”
关键： 这些范例不是从原始数据中取，而是 LLM 在元数据 + seed 上造出来的 synthetic dirty 数据 ，你可以和真实数据混合使用来做校准。

## 4. 新的 Calib & EM 层（基于 clean 规则 + synthetic dirty）
在新框架下，没有 dirty 规则，只有 clean rules；同时有一批“我们自己造的脏例”（synthetic dirty）。我们想要：

- 学出每条 clean rule 的可靠性参数；
- 给每个 cell 一个软分 S(i, j) ，表示它是 dirty 的概率。
### 4.1 数据视图
对每列 col ，我们有：

1. 真实数据中的 cell：
   - x_i = 真实值；
   - 一组 clean rule 输出： z_r(i) ∈ {0,1} （1=通过，0=未通过）。
   - 标签未知。
2. synthetic dirty：
   - 来自 DirtyExampleAgent.generate_dirty_examples() 的样本：
     - x̃_k = 合成的脏值；
     - 我们知道它是 dirty： ỹ_k = 1 ；
     - clean rules 对它的输出同样可以算 z_r(x̃_k) 。
再加上我们自己从真实数据中选的 clean seed：

- C ：高置信 clean 的真实 cell，认为 y=0 。
于是校准数据集分为三块：

- L_clean ： (x, z(x), y=0) 来自 clean seed；
- L_dirty ： (x̃, z(x̃), y=1) 来自 synthetic dirty；
- U ：未标注的真实 cell。
### 4.2 规则级参数：每条 clean rule 的 (α_r, β_r)
对每条 clean rule r：

- 定义：
  - α_r = P(z_r=1 | y=0) ：真干净通过该 clean 规则的概率（越接近 1 越好）；
  - β_r = P(z_r=1 | y=1) ：脏值仍通过该 clean 规则的概率（“漏报”率）。
- 直观：
  - 好的 clean rule 应该 α_r 高、β_r 低。
初始化：

- 用 L_clean 和 L_dirty 直接估计：
  - 在 clean seed 上： α_r ≈ (# z_r=1 on L_clean) / |L_clean|
  - 在 synthetic dirty 上： β_r ≈ (# z_r=1 on L_dirty) / |L_dirty|
### 4.3 EM 结构（简化版）
对 U 中的真实 cell（未标注）：

1. E-step：估计每个 cell 的脏概率 γ_i
   
   - 已知 rules 的 outputs z(i) ，以及规则参数 (α_r, β_r)；
   - 假设规则条件独立，做类似 naive Bayes 推断：
     
     ```
     P(y=0 | z(i)) ∝ π_0 * ∏_r P(z_r(i) | y=0)
     P(y=1 | z(i)) ∝ π_1 * ∏_r P(z_r(i) | y=1)
     
     其中：
     P(z_r=1 | y=0) = α_r，P(z_r=0 | y=0) = 1-α_r
     P(z_r=1 | y=1) = β_r，P(z_r=0 | y=1) = 1-β_r
     ```
   - 得到 γ_i = P(y=1 | z(i)) 。
2. M-step：使用 U 的软标签更新 (α_r, β_r) ：
   
   - 在 L_clean / L_dirty 之外，对 U 里的每个 cell i：
     
     - 看它的 rule 输出 z_r(i) ；
     - 使用 γ_i 对计数加权，更新“期望通过/未通过”的统计。
   - 综合 L_clean, L_dirty 以及 U 的期望，重新估计：
     
     ```
     α_r = 期望(z_r=1 & y=0) / 期望(y=0)
     β_r = 期望(z_r=1 & y=1) / 期望(y=1)
     ```
3. 迭代，直到收敛或达到 max_rounds 。
### 4.4 从参数到 cell-level score
对每个 cell i 最终我们有：

- γ_i = P(y=1 | z(i)) ：该 cell 是错误/脏的概率；
- 可以按 error family 拆解：
  - 对 missing：只用来自 family="missing" 的 clean rules的输出和参数；
  - 对 outlier：只用 family="outlier" 的规则；
  - 对 pattern：只用 family="pattern" 的规则。
产生：

- S_missing(i) = P(y_missing=1 | z_missing(i))
- S_outlier(i) = P(y_outlier=1 | z_outlier(i))
- S_pattern(i) = P(y_pattern=1 | z_pattern(i))
- S_total(i) 可以定义为这些的某种 max/sum。
这样就对应你 revision 里的 “Scoring & Fusion Agent：多规则软分数融合”。

## 5. 新的顶层流程（按列、无 cross-column、无旧模块依赖）
在此约束下，一个新的 run_clean_em_mode(args) 可以逻辑上长这样（伪代码级， 不涉及旧 conflict/gap 模块 ）：

1. 读数据 & profiling
   - 用 PandasProfiler(args.dirty_csv) 拿到 df + metadata 。
2. 按列生成 clean 规则
   - 对每列 col ：
     - 根据 metadata 的 type 调用：
       - CleanMissingAgent.generate_rules(col, metadata[col])
       - CleanOutlierAgent.generate_rules(col, metadata[col]) （仅 numeric / 或 category 特殊逻辑）
       - CleanPatternAgent.generate_rules(col, metadata[col])
     - 把所有 rule_str 编译成 rule_func ，填入 CleanRulePool[col] 。
3. 初始 clean seed 选择（简单共识）
   - 对每列，先用所有 clean rules 做一次打分：
     - 例如 score_clean(i, col) = (# 通过的 clean rules) / (# 参与投票的 clean rules) ；
   - 选取每列分数最高的前 p% 作为 L_clean[col] 的种子（高置信 clean）。
4. DirtyExampleAgent 生成 synthetic dirty
   - 对每列 col 、每个 family（missing/outlier/pattern）：
     - 从 L_clean[col] 抽样 seeds；
     - 调用 DirtyExampleAgent.generate_dirty_examples(...) 生成对应 family 的脏例；
     - 构成 L_dirty[col] 。
5. 规则输出矩阵计算
   - 对 L_clean[col] 、 L_dirty[col] 、以及全体真实 cell U[col] 计算所有 clean rules 的输出矩阵 z_r(i) 。
6. EM 校准
   - 对每列执行 EM，得到：
     - 每条规则的 (α_r, β_r) 参数；
     - 对所有 cell 的 γ_i （按 family 拆解得到 S_missing / S_outlier / S_pattern）。
7. 输出结果
   - 保存或打印每个 cell 的三个子分数 + 总分；
   - 选 top-K 高分错误候选用于可视化/下游。
整个过程中完全不依赖：

- Judge.evaluate_rules 的 old API；
- CleanRuleSet / DualRule / GapResolver / ConflictResolver ；
- 也没有 cross-column 规则/Agent 的参与。
## 6. 和你 4 点要求逐条对齐
1. “修改后规则只需要四个 Clean Agent，规则 True=正确，False=不正确”
   
   - 方案中所有规则都来自 CleanMissing/Outlier/Pattern/Relation（后者暂不实现），统一返回 True=正常；
   - 脏的定义统一为 “违反 clean 规则” + EM 中的 y=1 概念。
2. “脏数据生成由专门 agent，通过 LLM 生成符合现实的范例”
   
   - 设计了 DirtyExampleAgent ，只负责 dirty 例子的生成，不参与规则生成；
   - 使用 metadata + clean seeds，按 error family 生成 realistic dirty。
3. “放弃现有 conflict/gap resolve，不考虑与现有模块整合，设计全新框架（git 已保存）”
   
   - 新框架完全不使用 GapResolver / ConflictResolver / refine_clean_rules ；
   - 只复用 profiler / pattern explorer / safe_dict 等底层工具；
   - 规则精修的概念变成“EM 下的权重/可靠性学习”，而不是基于 conflict/gap 的 rule rewriting。
4. “暂时不考虑扩展点和 cross column 的错误类型和 Agent”
   
   - 当前设计中的 cross-column agent（CleanColRelationAgent）只是一个接口占位；
   - 具体实现和错误类型全部 deferred，不参与当前迭代的计算；
   - pipeline 中只对 per-column missing/outlier/pattern 做检测和校准。
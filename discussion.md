# Welkinni

2 那仔细思考一下，如果我需要在clean_em模式里加入reflection，可以怎么做呢？先仔细看一下代码，不需要修改，我们讨论一下；把 reflection 看成：在“calibration + EM”之前，用少量 局部 LLM rewrite 把明显不好的规则修一修，然后再进入现在的 clean_em 流程，就是用生成的dirty数据来反思一下规则？

# Builder with MCP
可以做，而且可以做到**很局部、很便宜**，并且复用你现有的组件（`RuleReviewerAgent`、`DirtyExampleAgent`、clean agents），整体上就是：

> 先用 clean rules + seeds + synthetic dirty 做一个“**小型监督评测**”，挑出明显坏的规则 → 用 LLM 做一次局部 rewrite → 只保留/替换那些通过简单指标验证“真的变好”的规则 → 然后再走现在的 calibration + EM。

我按“怎么插到现有 clean_em 里”来讲一版设计。

---

**1. 目标：在 clean_em 里局部加一层 reflection**

你给的定义很好：  
把 reflection 放在 “`clean_z / dirty_z` 计算完 → calibration + EM 之前”：

- 利用：
  - 高置信 clean seeds（近似 y=0）
  - DirtyExampleAgent 生成的脏样本（近似 y=1）
- 对每条 rule 做一个很小的“局部 A/B 测试”：
  - 如果 rule 在这两个集合上的表现很烂，就扔给 LLM 要一个 rewrite；
  - 新 rule 再用同样的 clean/dirty 集合测试一遍，确认“比老 rule 好”才收下；
  - 否则要么保持原样，要么直接 drop。

好处：

- 不改整个 clean_em 的理论框架，只是在 rule_pool 上做一次 **self-healing**；
- 只在“已经有明显信号的坏规则”上调用 LLM，成本可控。

---

**2. clean_em 里你已经有的“监督信号”**

在当前 `run_clean_em_mode` 的 family 循环里，每个 family 已经算了：

- `clean_z`：shape `[n_clean_seeds, n_rules_family]`  
- `dirty_z`：shape `[n_dirty_examples, n_rules_family]`

对第 r 条规则的两列向量：

- `clean_col = clean_z[:, r]`，`dirty_col = dirty_z[:, r]`  

可以直接得到四个基本统计：

- `clean_pass = clean_col.mean()`：该 clean 规则在 clean seeds 上返回 True 的比例；
- `dirty_pass = dirty_col.mean()`：在 dirty examples 上返回 True 的比例；
- `clean_viol = 1 - clean_pass`：在干净上 false 的比例（假阴性）；
- `dirty_viol = 1 - dirty_pass`：在脏上 false 的比例（你可以看作“检出率”）。

在我们刚改完的语义下：True = 干净，所以一个“好规则”的期望是：

- `clean_pass` 高，`dirty_pass` 低。
- 反过来：
  - `clean_pass` 很低 → 这条 clean 规则在真正干净数据上经常拒绝，是“过于苛刻”；
  - `dirty_pass` 很高 → 在脏样本上也经常说“干净”，没区分力。

这已经是一个很好的**局部监督信号**，完全可以驱动 reflection 的触发 & 验证。

---

**3. 触发 reflection 的“候选规则选择”**

为了控制 LLM 调用量，可以只对“明显差”的规则做 rewrite 尝试，例如：

- 按 family 先筛一遍：

  ```text
  if clean_pass < bad_clean_threshold
     or dirty_pass > bad_dirty_threshold:
      → 标记为 reflection 候选
  ```

- 阈值可以比 calibration 更宽松，例如：
  - `bad_clean_threshold = 0.6`
  - `bad_dirty_threshold = 0.6`
- 再加几个 guard，避免噪声：
  - dirty 样本数太少的 family 先不要反思；
  - 每列每个 family 最多反思前 k 条最差的规则（按某个 score 排序）。

这样每个 column × family 可能只触发 0–几次 reflection，成本就很小。

---

**4. 反思时给 LLM 的上下文**

你已经有 `RuleReviewerAgent`，虽然它现在是为 dual 设计的，但思路可以照抄：

对某个待反思的 clean rule，构造一个简化版的 prompt，内容可以包括：

- 列信息：
  - column 名、type、重要统计（min/max/unique/top_values 等），直接从 `metadata[column]` 来；
- 当前 clean rule：
  - 以“**定义正常值**”的语义描述：
    - “当前规则是一个 clean predicate：返回 True 表示值正常；False 表示值异常。”
- 正/反例 sample（这是最关键的）：
  - 从 clean seeds 中抽几条：
    - `clean_col == 0` 的样本：**应该是干净，但你 rule 判成 False** 的例子；
    - 这些用 `label="clean"` 标记；
  - 从 synthetic dirty 中抽几条：
    - `dirty_col == 1` 的样本：**应该是脏，但你 rule 判成 True** 的例子；
    - 这些用 `label="dirty"` 标记；
- 任务要求：
  - 让 LLM“修正规则”，减少上述两类错误；
  - 保持 True=正常 的语义；
  - 返回一个新的 `lambda value, row=None: ...`。

这一块其实跟 `RuleReviewerAgent.suggest_rule_fix` 很像，只是 context 换成了：

- “clean rule + clean/dirty pseudo labels”，而不是 “P_clean + P_dirty + conflict/gap samples”。

---

**5. 如何验证 rewrite 的好坏：一个局部 A/B test**

拿到 LLM 修过的 `new_rule_str` 之后，你可以做一个**非常本地的评估**：

1）试着 `eval(new_rule_str, safe_dict)`：

- 编译失败 → 直接丢弃这个 rewrite，保留旧 rule 或干脆 drop。

2）在相同的 clean/dirty 集合上重算通过率：

- `new_clean_pass` = 在 clean seeds 上新 rule 返回 True 的比例；
- `new_dirty_pass` = 在 dirty examples 上新 rule 返回 True 的比例。

3）用一个简单的 objective 来判断是否接受新规则，例如：

- 定义一个“损失”：

  ```text
  L_old = w1 * (1 - clean_pass) + w2 * dirty_pass
  L_new = w1 * (1 - new_clean_pass) + w2 * new_dirty_pass
  ```

  - 这里 `w1, w2` 可以一样，也可以稍微偏向 recall（降低 dirty_pass）；
- 若 `L_new + margin < L_old`，且 `new_clean_pass >= min_clean_floor`，就接受 new rule：
  - 在 rule_pool 里把这一条的 `rule_str/func` 替换成新的；
- 否则：
  - 要么保持旧 rule；
  - 要么干脆标记“坏到不可救药”直接 drop（比如 `clean_pass` 极低而又修不回来）。

这一步的好处是：

- 完全“自洽”：只用你已有的 pseudo labels（clean seeds + dirty examples）；
- 无需 ground truth；
- 不需要复杂 search，只是“现有 rule vs LLM 一次 rewrite”的单步比较。

---

**6. 和现有 clean_em 的关系**

把以上步骤插入到你现在的流程里，大致是：

1）已有流程（每列 × family）：

- 生成 clean rules → `rule_pool`；
- 计算 `outputs` / `clean_z` / `dirty_z` / `unlabeled_z`。

2）新增：**reflection block**（只在这个 family 内）：

- 依据 `(clean_pass, dirty_pass)` 挑出若干“坏规则”；
- 对这些规则逐个调用 “RuleReviewer-like agent” 生成 `new_rule_str`；
- 对每个 new rule 做局部 A/B test，决定：
  - 接受 rewrite（更新 rule_pool + 相关列到 `clean_z/dirty_z/unlabeled_z`）；或
  - 保留旧 rule；或
  - 直接 drop。

3）然后才是你现在已经有的：

- rule-level calibration（我们刚改成 True=干净的版本）；
- EM；
- 汇总 `gamma → S_missing/S_outlier/S_pattern → S_total`。

也就是说：

> clean_em 不需要变成一个复杂的多轮 refinement，只需在「进入 calibration+EM 前的这一小块」加一个“单步 self-healing”。

---

**7. 总结一下设计要点**

- **触发条件**：用 `clean_pass` 和 `dirty_pass` 识别“明显差”的规则；
- **数据来源**：仅用 high-confidence clean seeds 和 DirtyExampleAgent 生成的 dirty examples；
- **反思主体**：复用 `RuleReviewerAgent` 的思路，写一个专门的 clean-rule rewriter（可以是新的 agent，也可以扩展现有的）；
- **验证机制**：在同一批 clean/dirty 样本上做局部 A/B test，只有“指标确实改善”的 rewrite 才生效；
- **插入位置**：放在当前 family loop 里 `clean_z/dirty_z` 计算之后、calibration+EM 之前。

如果你愿意，我们下一步可以选一个具体列（比如 flights 里的某个 numeric 列），手工走一遍“发现坏规则 → 构造 prompt → 设计 accept 条件”的完整例子，把这套 reflection workflow具体化成一个函数签名和伪代码。


# Welkinni

2 可以的，来实现一下，就是在反思时给 LLM 的上下文中，当前clean rule可以给出具体的形式吧

# Builder with MCP
toolName: todo_write
            
status: success
          
Todos updated: 2 items
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/agent.py
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/agent.py
          
changes: 
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/main.py
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/main.py
          
changes: 
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/main.py
          

        
            
toolName: apply_patch
            
status: failed
          
          

        
            
toolName: edit_file_update
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/main.py
          
changes: 
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 2 items
          
已经在 clean_em 里实现了一层“基于 dirty 样本的局部 rule reflection”，而且在给 LLM 的 prompt 里会把当前 clean rule 的具体形式直接塞进去一起考虑。

下面按“做了什么”和“数据怎么流动”简单讲一下。

---

**1. 新增的反思 Agent：CleanRuleReflectionAgent**

在 [agent.py](file:///mnt/data/welkinni/table_det/agent.py#L1222) 里，我加了一个专门用于 clean 规则反思的 agent：

- 类名：`CleanRuleReflectionAgent(BaseAgent)`
- 关键方法：`refine_clean_rule(...)`

签名：

```python
def refine_clean_rule(
    self,
    column: str,
    metadata: Dict[str, Any],
    family: str,
    rule_str: str,
    clean_mis_examples: List[Dict[str, Any]],
    dirty_mis_examples: List[Dict[str, Any]],
) -> Optional[str]:
```

它做的事：

- 构造 prompt，内容包括：
  - 列名：`Column: {column}`
  - 列类型：`Column type: {column_type}`
  - 错误 family：`Error family: {family}`（missing / outlier / pattern）
  - 当前 clean rule 的完整字符串：
    - `Current clean rule (returns True for normal/clean values): {rule_str}`
  - 在 clean seeds 中被错杀的样本：
    - `Clean examples that SHOULD return True but currently return False:`
    - 一个 JSON 数组（最多 10 条）：`[{"value": ...}, ...]`
  - 在 synthetic dirty 中被误放行的样本：
    - `Dirty examples that SHOULD return False but currently return True:`
    - 一个 JSON 数组（最多 10 条）：`[{"value": ..., "reason": ...}, ...]`
  - 代表性的高频值：`top_values`（方便 LLM理解分布）
  - 任务要求：
    - 保持 True=干净 语义；
    - 对 clean 样本更宽松，对 dirty 样本更严格；
    - 处理 None/NaN/空串；
    - 最终只返回一个 `lambda value, row=None: <expr>`。

- 调用 `_call_llm(...)`，然后用 `_extract_lambda` 把第一行 lambda 抽出来作为 `new_rule_str` 返回。

这样，你说的“当前 clean rule 可以给出具体的形式”已经体现在 prompt 里了。

---

**2. 在 clean_em 里如何利用这层 reflection**

在 [main.py 的 run_clean_em_mode](file:///mnt/data/welkinni/table_det/main.py#L296-L488) 里，我在 family 循环中插入了反思逻辑：

1）初始化 agent（在 EM 大循环前）：

```python
dirty_agent = DirtyExampleAgent(base_url=args.base_url, model=args.model)
reflection_agent = CleanRuleReflectionAgent(base_url=args.base_url, model=args.model)
```

2）原有流程保持：  
对每个 column × family：

- 得到：
  - `clean_z`: seeds 上 per-rule 输出；
  - `unlabeled_z`: 未标注行上 per-rule 输出；
  - `dirty_z`: synthetic dirty 上 per-rule 输出；
- 以及 seeds 的真实值 `seeds_values` 和 dirty examples 列表 `dirty_examples`。

3）在“calibration + EM”之前新增 reflection block：

- 先算每条规则的通过率：

  ```python
  clean_pass = clean_z.mean(axis=0)
  dirty_pass = dirty_z.mean(axis=0)
  ```

- 用一个简单的标准挑出“坏规则”做候选：

  ```python
  bad_clean_threshold = 0.6
  bad_dirty_threshold = 0.6
  if cp < bad_clean_threshold or dp > bad_dirty_threshold:
      进入候选集合
  ```

  并限制每个 family 最多反思 `max_reflections = 3` 条规则。

- 对每个候选规则：

  1. 找出它在 clean / dirty 上“犯错”的样本：
     
     ```python
     clean_mis_indices = np.where(clean_z[:, local_idx] == 0)[0]  # seeds里被错杀
     dirty_mis_indices = np.where(dirty_z[:, local_idx] == 1)[0]  # dirty里被误放行
     ```

     然后构造传给 LLM 的示例：

     ```python
     clean_mis_examples = [{"value": seeds_values[i]} ...]
     dirty_mis_examples = [{"value": ex["value"], "reason": ex["reason"]} ...]
     ```

  2. 调 `reflection_agent.refine_clean_rule(...)`，把：
     - `column`
     - `metadata[column]`
     - `family`
     - 当前 `rule_str`
     - 上述 clean/dirty 反例
     
     一起塞进 prompt，请 LLM 给出新的 lambda：

     ```python
     new_rule_str = reflection_agent.refine_clean_rule(...)
     ```

  3. 用 `eval(new_rule_str, safe_dict)` 编译，如果失败就跳过。

  4. 在相同的数据上做一个局部 A/B test：

     - 用新 rule 在整列上重新算一遍输出：

       ```python
       new_all = [ new_rule_func(value, row) for row in df[column] ]
       new_clean_col = new_all[seed_mask]
       new_unlabeled_col = new_all[~seed_mask]
       new_dirty_col = [ new_rule_func(val, fake_row) for val in dirty_vals ]
       ```

     - 得到新的通过率：

       ```python
       new_cp = new_clean_col.mean()
       new_dp = new_dirty_col.mean()
       ```

     - 定义一个很简单的“损失函数”来衡量好坏：

       ```python
       old_loss = (1 - cp) + dp
       new_loss = (1 - new_cp) + new_dp
       ```

     - 接受条件：

       ```python
       if new_loss <= old_loss - improve_margin and new_cp >= args.calib_min_clean_pass:
           接受新rule
       ```

       其中 `improve_margin = 0.02`。

  5. 如果接受：

     - 更新这一条规则在所有地方的表现：

       ```python
       clean_z[:, local_idx]     = new_clean_col
       unlabeled_z[:, local_idx] = new_unlabeled_col
       dirty_z[:, local_idx]     = new_dirty_col
       rules[rule_idx]["rule_str"]  = new_rule_str
       rules[rule_idx]["rule_func"] = new_rule_func
       clean_pass[local_idx] = new_cp
       dirty_pass[local_idx] = new_dp
       ```

     - 打一条 log，记录前后 clean_pass / dirty_pass 的变化。

4）之后的逻辑不变：

- 用更新后的 `clean_pass / dirty_pass` 做现在的 calibration：

  ```python
  keep_mask = (clean_pass >= args.calib_min_clean_pass) & (
      dirty_pass <= args.calib_max_dirty_pass
  )
  ```

- 丢掉不满足门槛的规则；
- 用剩下的规则 + `clean_z / dirty_z / unlabeled_z` 进 EM；
- 最终得到 `gamma → S_missing/S_outlier/S_pattern → S_total`。

---

**3. 总体直觉**

现在的 clean_em pipeline 变成：

- Clean agents 生成 clean rules；
- 在真实数据 + seeds + synthetic dirty 上，先对每条 clean rule 做一个“小型 supervised 评测”；
- 对明显差的 rule：
  - 把 **当前 lambda 文本 + 干净/脏反例** 一起丢给 LLM；
  - 要一个新的 lambda，要求 True=clean/False=dirty；
  - 在同一批数据上做 A/B 对比，不明显变好就不用；
- 把“自修过”的规则再进入 `calibration + EM`，做最后的筛选和弱监督推断。

你如果想看某一条具体规则的反思过程，可以跑一次 clean_em，看 `CleanEM` 日志里：

- `Reflecting rule 'xxx' before calibration (...)`
- `Refined rule 'xxx': clean_pass a->b, dirty_pass c->d`

来 inspect 它在 prompt 里看到的旧 rule 和更新后的 lambda。


# Welkinni

2 20260209_173247_hospital_error-01_cleanem.log 那仔细分析一下目前CountyName列上的log还有对应的数据，为什么现在性能这么差呢？很奇怪啊，深入一点分析

# Builder with MCP
toolName: search_by_regex
            
status: success
          
query: CountyName
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log
          

        
            
toolName: view_folder
            
status: success
          
/mnt/data/welkinni/table_det/data
          

        
            
toolName: search_by_regex
            
status: success
          
query: hospital_error-01
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/data/hospital_error-01.csv
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/data/hospital_clean.csv
          
先把 CountyName 这列的行为从 log + 数据里拆出来看一下，会比较清楚为什么现在是“Recall=1.0 + 超多 false positive”。

---

**1. 从整体指标看 CountyName 问题出在哪**

log 末尾的 per-column 指标里，CountyName 是这样的（[log](file:///mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log#L1581-L1584)）：

- CountyName:
  - precision ≈ 0.117
  - recall = 1.0
  - f1 ≈ 0.209
  - true_positives = 60
  - false_positives = 453
  - false_negatives = 0
  - total_ground_truth_errors = 60
  - total_detected_errors = 513

也就是说：

- 这列确实把所有 60 个真错误都抓到了（FN=0），**Recall 完美**；
- 但同时把 453 个干净 cell 也误报成错了，**FP 很大**。

结合 CountyName 的规则和 EM 输出，可以理解成：  
CountyName 上的 `S_total` 对大部分行都给了比较高的分数，导致 threshold=0.5 时很多正常县名也被判为错误。

---

**2. CountyName 的规则长什么样**

log 中 CountyName 的规则池（[log](file:///mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log#L110-L115)）：

- 总共 11 条规则：
  - family=missing: 1 条 completeness
  - family=outlier: 5 条 accuracy
  - family=pattern: 5 条 pattern_consistency

关键几条（我简化一下）：

- missing:

  ```python
  completeness_0:
    lambda value, row=None: (
      value is not None and
      str(value).strip().lower() not in ['', 'empty', 'n/a', 'na', 'nan', 'none', 'null', 'unknown', 'xxxxx']
    )
  ```

- outlier:

  ```python
  accuracy_1: isinstance(value, str) and len(value) > 0 and len(value) <= 50
  accuracy_2: value.lower() in [jefferson, etowah, marion, ...]   # top10 county
  accuracy_3: bool(re.match(r'^[a-zA-Z\s]+$', value))
  accuracy_4: not any(char.isdigit() for char in value)
  accuracy_5: value.strip().isalpha() or value.strip().replace(' ', '').isalpha()
  ```

- pattern:

  ```python
  pattern_6:  ^[a-zA-Z]{3,10}$  (只字母、长度3-10)
  pattern_7:  len in {3,4,6,7,8,9,10}
  pattern_8:  str(value).strip().isalpha()
  pattern_9:  value in [jefferson, etowah, ...]  # top10
  pattern_10: not any(char.isdigit() for char in str(value).strip())
  ```

对 CountyName 这种列来说，**绝大多数真实值都是干净的纯字母英文县名**（比如 dirty/clean 文件里都只有 `jefferson`, `houston`, ...），所以这些 clean 规则在真实数据上几乎全部返回 True。

---

**3. CountyName 上 seeds + dirty examples + calibration + EM 的具体行为**

从 log 中 CountyName 部分（[log](file:///mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log#L671-L737)）：

1）**Clean seeds 选择**

- Clean seeds：100 行（10%），`Clean Z shape: (100, k)`；
- 因为规则都比较“宽松”，在干净数据上几乎都返回 True。

2）**missing family**

- 使用 rule completeness_0；
- dirty_missing 由 `DirtyExampleAgent` 生成：`['', 'None', ' ', 'not specified', '-']`
- 结果：

  ```text
  clean_pass = 1.000
  dirty_pass = 0.400
  decision = DROP
  All rules rejected for family 'missing'
  ```

直觉：  
这个 completeness 规则在 dirty_missing 的样本上，有 40% 的“脏值”仍判为干净（主要是 `'not specified'`、`'-'` 还没覆盖到），所以被校准逻辑直接扔掉了；  
结果是 **CountyName 完全没有 missing family 的 score**。

3）**outlier family**

- 使用 5 条 accuracy 规则；
- Dirty outlier examples：`['12345', 'houston123', 'verylongcountynameindeed', 'short', 'HouStOn', '...']`

log 显示反思+校准之后：

- 反思前：

  - 部分规则在 dirty 上通过率很高，比如 `accuracy_1` 初始 dirty_pass=1.0（所有脏样本都被认作“clean”）。

- 反思：

  ```text
  Reflecting accuracy_1 (clean_pass=1.000, dirty_pass=1.000)
    → clean_pass 1.000 -> 0.960, dirty_pass 1.000 -> 0.167

  Reflecting accuracy_3 (clean_pass=1.000, dirty_pass=0.667)
    → clean_pass 1.000 -> 1.000, dirty_pass 0.667 -> 0.167

  Reflecting accuracy_4 (clean_pass=1.000, dirty_pass=0.667)
    → clean_pass 1.000 -> 1.000, dirty_pass 0.667 -> 0.333
  ```

  说明：

  - LLM 确实把几条太宽的 rule 收紧了一些，让它们在 dirty 示例上 False 多一点。

- 校准（门槛是 min_clean_pass=0.8, max_dirty_pass=0.3）：

  ```text
  accuracy_1: clean_pass=0.960, dirty_pass=0.167 → KEEP
  accuracy_2: clean_pass=1.000, dirty_pass=0.167 → KEEP
  accuracy_3: clean_pass=1.000, dirty_pass=0.167 → KEEP
  accuracy_4: clean_pass=1.000, dirty_pass=0.333 → DROP
  accuracy_5: clean_pass=1.000, dirty_pass=0.667 → DROP
  ```

- EM 输出：

  ```text
  gamma_mean = 0.5779
  gamma_std  = 0.4858
  range      = [0.0001, 0.9988]
  High scores (>0.5): 513 / 900
  Alpha (on clean)  ≈ [0.906, 0.999, 0.999]
  Beta  (on dirty)  ≈ [0.0019, 0.0166, 0.9905]
  ```

这里有两个非常重要的信号：

- 有 3 条 outlier-clean 规则通过了校准，**但其中至少有一条在 dirty 上 recall 很高（beta≈0.99）**；
- EM 的 gamma 分布在 unlabeled 上：
  - 平均 0.58，std 大，范围 [~0, ~1]；
  - 有 513/900 个 unlabeled cell 的 gamma > 0.5。

再看 CountyName ground truth：

- 总共 1000 行；
- 真错误 60 个 → 都是错误的县名 / 拼写。
- 但 EM 下有 513 个 cell 被 family-outlier 判为 gamma>0.5（再加上 pattern family 也有类似的高 gamma），导致 S_outlier/S_pattern/S_total 对大量正常 county 也很高。

4）**pattern family**

pattern family 的行为很像 outlier：

- 保留了 3 条 pattern 规则（包括“在 top10 县名列表里”这种）；
- EM 输出也差不多：

  ```text
  gamma_mean = 0.5765
  high_scores (>0.5): 513 / 900
  beta: [0.94, 0.014, 0.0019]
  ```

说明 pattern family 里也有一条 rule 在 dirty 示例上非常敏感（beta≈0.94）。

因为 final `S_total` 是三个 family `max(S_missing,S_outlier,S_pattern)`，而 CountyName 没有 missing family，基本就是 max(outlier, pattern)。  
outlier + pattern 两个 family 都学到“**大约一半以上的 unlabeled cell 是错误**”，而真实错误只有 60/1000=6%。  
这就很自然导致：

- 所有真错（6%）都高分 → Recall=1；
- 还有大约 450 个干净 cell 也高分 → Precision 很低。

---

**4. 为什么 EM 会认为这么多县名是“脏”的？**

核心原因是：  
CountyName 这个列的 **dirty 示例 / clean seeds / rule设计** 组合起来，对 EM 来说形成了一个“非常有偏”的监督信号。

几点具体问题：

1）**dirty 示例数量太少且分布太窄**

- CountyName dirty-outlier 的例子只有 6 个，dirty-pattern 只有 10 个：
  
  ```text
  outlier dirty values: ['12345', 'houston123', 'verylongcountynameindeed', 'short', 'HouStOn', ...]
  pattern dirty values: ['houstn', 'cofee', 'Houston', 'hous ton', 'houston123', ...]
  ```

- 它们覆盖的是极端情况：有数字、拼写错误、大小写/空格混乱、长度极端等；
- 但你真实数据里的错误 county 只占 6%，且可能并不完全长得这么极端。

EM 的 generative 假设相当于：

> “这些 dirty 示例代表了所有错误 county 的分布”。

在这么少、这么偏的 dirty 示例上估计出的 beta，很容易让某些规则看起来“对 dirty 很敏感”，进而在 unlabeled 上推高大量 cell 的 γ。

2）**clean seeds 的选择方式也有偏差**

- seeds 是根据 clean rules 对真实数据的通过率选 top 10%；
- CountyName 上 clean rules 其实非常宽松（长度、字母、top10 列表等），对错拼和脏值也可能返回 True；
- 如果 ground truth 的错误 county 只是非常少数的拼写轻微错误，clean rules 很可能也给它们打高分，导致：
  - 这些错误被误当成 seeds；
  - 于是 `clean_z` 里混入了一些“脏但被当作干净”的样本；
  - EM 会把这部分“脏但被 clean rule 认为是 True”的行为解释为：**这条 rule 在 y=0（干净）下的 True 率很高**，从而推高 alpha，削弱这条 rule 的 diagnostic 力。

3）**规则语义是 true=clean，但用于 outlier family**

看 CountyName 的 outlier/pattern 规则，其实大部分是：

- “是字符串、长度合理”；
- “只含字母和空格”；
- “不含数字”。

这些规则在正常县名上几乎都是 True，在轻微拼写错误上大概率也 True。  
反而真正敏感的是那种 “value 在 top10 县名列表里” 的规则，这条 rule 在正确的 top10 县名上 True，在很多错误拼写上 False；  
但它在 dirty 示例上可能并不那么突出（因为 dirty示例数量少且构造方式与真实错误不完全一致）。

综合起来导致：

- EM 容易学到：“只要不是极端 dirty 示例，很多 rule 输出模式在 clean / dirty 之间并不显著不同”；
- 在这种弱区分度下，再加上先验 `prior_dirty=0.05`，EM 可能在 unlabeled 上把整个后验拉向中间甚至偏高一点（因为少量 strong rule 的 beta 高会推高一大片样本的 γ）。

4）**missing family 被完全丢弃**

CountyName 中实际的错误类型很可能包括 missing/placeholder（例如空字符串、特殊 token），但：

- completeness_0 在 synthetic missing 示例上只拒绝了 60%（dirty_pass=0.4），被 calibration 认为“不够好”直接 drop；
- 结果 CountyName 完全没有从 missing family 获得“强 signal”，全部压力落在 outlier/pattern 上。

---

**5. 为什么表现会“感觉很奇怪”**

组合一下：

- 真实错误比例很低（6%，60/1000）；
- EM 却在 CountyName 的 outlier/pattern 两个 family 上都得到：
  - `gamma_mean ≈ 0.58`；
  - `>0.5` 的 cell 有 513/900；
- 这意味着：
  - 在这两个 family 的 generative 模型里，EM 认为“错误县名的比例接近一半”，
  - 并且有一条 rule 对 dirty 示例的 beta≈0.99（almost always True under y=1）；
- 最终 threshold=0.5 时，S_total>=0.5 的 cell 数量大概就是 500+，跟 log 里的 513 一致。

也就是说：

> 对 CountyName 来说，EM 的“世界观”与真实世界完全不匹配：  
> 它以为这个列有一半是错的，而真实只有 6%。

这是“弱监督 + 偏样本”的典型问题：dirty 示例过窄，clean seeds 也可能被污染，导致 EM 在未标注区间 over-estimate 错误比例。

---

**6. 给你一个简短结论**

综合 log 和数据，现在 CountyName 性能差（精度很低）的原因主要是：

- 规则设计本身相当宽泛，很多规则对轻微错误几乎不敏感；
- synthetic dirty 示例数量少且偏极端，导致 EM 学到的 rule 行为在真实错误分布上并不可靠；
- clean seeds 由这些宽松规则选出，很可能把部分错误也当成 seeds，污染了 `clean_z`；
- missing family 的 completeness 规则被 calibration 丢掉，使得 CountyName 只能靠 outlier/pattern 两个 family；
- 在这样的弱监督条件下，EM 把“错误比例”估计得太高，导致大约一半的 CountyName cell 被判为错误 → Recall=1.0 但 Precision 很差。

如果你愿意，下一步我们可以专门针对 CountyName 设计一个更稳的策略，比如：

- 把 CountyName 的 dirty 示例改成基于真实地名词典 + 拼写扰动；
- 对 CountyName 单独调低 `prior_dirty`；
- 或者对 CountyName 在 EM 之后加一个 column-specific threshold（>0.5 不够，要 >0.8 才当错误），专门降 FP。

# Welkinni

2 main.py 那如果深入看一下clean_em的方案，结合log在方案每个步骤的记录，分析一下为什么会导致目前的问题呢？



# Builder with MCP

toolName: CompactFake
            
status: success
          
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 1 items
          

        
            
toolName: view_folder
            
status: success
          
/mnt/data/welkinni/table_det/data
          

        
            
toolName: view_folder
            
status: success
          
/mnt/data/welkinni/table_det
          

        
            
toolName: file_search
            
status: success
          
file_pattern: **/*
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log
          

        
            
toolName: search_by_regex
            
status: success
          
query: CountyName
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log
          

        
            
toolName: file_search
            
status: success
          
file_pattern: **/hospital_error-01.csv
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/data/hospital_error-01.csv
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173714_hospital_error-01_clean_em_scores.csv
          

        
            
toolName: search_by_regex
            
status: success
          
query: ,CountyName,
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173714_hospital_error-01_clean_em_scores.csv
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/label_model.py
          

        
            
toolName: search_codebase
            
status: success
          
query: implementation of EM in clean_em pipeline, gamma_mean logs
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/main.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/data/welkinni/table_det/main.py
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 1 items
          
**结论先说**

CountyName 这一列现在的行为基本是：  
“把绝大多数行都当成错误”，所以 recall=1.0、但 false positive 非常多（453/513），本质原因是：

- 规则设计把“最常见的 10 个县名”几乎当成“唯一正常值”
- clean seed 选择强烈偏向这 10 个高频县
- synthetic dirty 的分布又主要在这几个高频县周围
- EM 在这样的先验 + 规则统计下会认为：**只要 CountyName 不在 top10，就很像错误**

这三者叠加起来，把“稀有但正确的县名”全部推成了 error，所以 CountyName 列整体被严重高估为“脏”。

下面按 log + 数据，把链条拆开说明。

---

**1. 先看 CountyName 的规则长什么样**

在 clean_em log 里，CountyName 对应的 clean 规则是（行号约 110–121）：  
[main.py 生成规则的 log 片段](file:///mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173247_hospital_error-01_cleanem.log#L110-L121)

关键几条：

- `accuracy_1`（outlier family）  
  `len(value) > 0 and len(value) <= 50`  
  - 非空且长度不超过 50，几乎所有条目都通过
- `accuracy_2`（outlier family）  
  `value.lower() in [ "jefferson","etowah","marion","marshall","covington","coffee","montgomery","houston","morgan","calhoun" ]`  
  - “是否属于前 10 高频县名”的白名单
- `accuracy_3` / `accuracy_4` / `accuracy_5`：  
  - 只允许字母 + 空格、不包含数字、`isalpha` 等
- pattern family 也有类似两条“top10 白名单 + 无数字”的规则：
  - `pattern_consistency_9`: `value in ['jefferson', ..., 'calhoun']`
  - `pattern_consistency_10`: `not any(char.isdigit() for char in value)`

关键点：**有两套“属于 top10 县名”的规则（accuracy_2 + pattern_9），它们在后面会变成非常强的信号**。

---

**2. clean seeds 选得很偏：主要是高频县**

在 CountyName 段落（行 671–674）：

- Clean seeds 选了 100 行（10%）：
  - `Seed indices (first 10): [37, 38, 39, 968, 969, 970, 971, 972, 29, 30]`

结合数据 [hospital_error-01.csv](file:///mnt/data/welkinni/table_det/data/hospital_error-01.csv#L2-L66) 看：

- 前几十行里 CountyName 基本是 `jefferson`, `houston`, `marshall` 这些**高频县**
- seed 里出现的 index 29,30,37,38,39 等，恰好都是这些医院、这些县名对应的多行记录

也就是说：

- CountyName 的 clean seeds 几乎全是“top10 高频县”的干净值
- 像 `lauderdale`, `crenshaw`, `de kalb` 这种**低频但正确的县**在 seeds 里几乎看不到

在 EM 模型的统计里，这会让“属于 top10”这个特征，在 clean 部分看起来几乎是“必然成立的条件”。

---

**3. synthetic dirty 的分布：只在高频县附近乱改**

看 CountyName 段落里的 synthetic dirty 例子（行 691–693、720–722）：

- outlier family 生成的脏例子：
  - `['12345', 'houston123', 'verylongcountynameindeed', 'short', 'HouStOn', ...]`
- pattern family 生成的脏例子：
  - `['houstn', 'cofee', 'Houston', 'hous ton', 'houston123', ...]`

特点：

- 大部分脏例还是围绕 `houston`、`coffee` 等高频县改出来的
- 很少 / 没有“低频但正确”的县，例如 `lauderdale`, `de kalb`, `crenshaw` 之类
- 也没有专门造“真实世界中常见的县名错拼写”，更多是简单的大小写/字符改动

这直接影响了规则的初始 α/β 估计：

在 `_run_em_for_family` 里（[main.py:202-247](file:///mnt/data/welkinni/table_det/main.py#L202-L247)）：

- α_r 初始：clean seeds 上规则输出的平均值
- β_r 初始：synthetic dirty 上规则输出的平均值

由于：

- seeds 只覆盖 top10 县
- synthetic dirty 也主要是从 top10 上改出来的错拼

所以像 “属于 top10” 这种规则，在这两个集合里看起来几乎是：

- clean 上：**几乎总是 True**（α ≈ 1）
- dirty 上：**几乎总是 False**（β ≈ 0）

也就是：在模型眼里，  
**“属于 top10 县名” ≈ “几乎必然是干净”；反过来，“不属于 top10 县名” ≈ “几乎必然是错误”**。

---

**4. 校准阶段：哪些规则被保留？**

CountyName 部分的校准 log（行 682–705, 724–730）：

- missing family：
  - completeness_0：`clean_pass=1.000, dirty_pass=0.400, decision=DROP`
  - 所以“缺失检测”的规则被直接丢掉，后续不参与 EM。
- outlier family：
  - 保留：accuracy_1, accuracy_2, accuracy_3  
    - accuracy_1：`clean_pass=0.960, dirty_pass=0.167, KEEP`  
    - accuracy_2：`clean_pass=1.000, dirty_pass=0.167, KEEP`（top10 白名单）
    - accuracy_3：`clean_pass=1.000, dirty_pass=0.167, KEEP`（只含字母/空格）
  - 丢弃：accuracy_4, accuracy_5（也是“不含数字/只含字母”的变体）
- pattern family：
  - 保留：pattern_6, pattern_9, pattern_10  
    - pattern_6：`^[a-zA-Z]{3,10}$`
    - pattern_9：top10 白名单
    - pattern_10：不含数字
  - 丢弃：pattern_7, pattern_8

也就是说：

- **最终进 EM 的，是 6 条非常宽松/频率驱动的规则，其中两条是“top10 白名单”**

按照当前阈值（`calib_min_clean_pass=0.8`, `calib_max_dirty_pass=0.3`），这些规则表现都“挺好”，所以全部被保留。但这种“好”是对 seeds + synthetic dirty 而言，对真实数据未必。

---

**5. EM 学出来的参数：top10 白名单变成了极强的“错误指示器”**

CountyName 的 EM 结果（行 707–711, 732–736）：

- outlier family：
  - `gamma_mean=0.5779, ... High scores (>0.5): 513/900`
  - Alpha（清洁上的通过率）：`[0.906, 0.999, 0.999]`
  - Beta（脏数据上的通过率）：`[0.00194, 0.01657, 0.99050]`
- pattern family：
  - `gamma_mean=0.5765, ... High scores (>0.5): 513/900`
  - Alpha：`[0.999, 0.9987, 0.9040]`
  - Beta：`[0.9395, 0.0141, 0.00192]`

结合规则来看：

- 对 outlier family：
  - 某条规则（对应 Beta≈0.01657，多半是 accuracy_2 / top10 白名单）：  
    - 对错误几乎从不通过（β≈0.016，也就是 98% 错误会 fail）
    - 对干净 α≈1
  - 这意味着：一旦某行在这条规则上 `z=0`（不在 top10），  
    - P(z | y=0) 非常小  
    - P(z | y=1) 非常大  
    ⇒ 对 EM 来说，这是“强烈支持 y=1（脏）”的证据。
- pattern family 中的 pattern_9（也是 top10 白名单）类似：Beta≈0.0141。

因为：

- 真实数据里，CountyName 的 **ground truth 错误只有 60 行**（log 末尾的 metrics：true_positives=60）
- 但数据里存在大量正确的低频县（`lauderdale`, `crenshaw`, `de kalb` 等）

对于这些低频且正确的县：

- seed 里几乎没见过这些值，所以 “top10 白名单” 的 α 被估得特别高
- synthetic dirty 也没把它们作为“正常样本”来看，β 仍然很低
- 结果：在真实未标注数据 U 上，只要 CountyName 不在 top10，这两条规则就呈现为“强烈错误信号”

这就是为什么 EM 的 gamma_mean > 0.57，而且有 513/900 个未标注 cell 的 gamma>0.5：  
**任何不在 top10 的县名行，都被模型当成“很像错误”的行。**

---

**6. 看一下 scores CSV：干净/脏的 CountyName 都被打成 ~0.999**

在 scores 文件里（[clean_em_scores.csv](file:///mnt/data/welkinni/table_det/results/agentic_error_detector/20260209_173714_hospital_error-01_clean_em_scores.csv)），搜索 `CountyName`：

部分条目是：

- 行 2015–2035：
  - `row_index=219..243, CountyName,de kalb / dx kalb / xe kalb ...`
  - `S_missing=0.0, S_outlier≈0.9988, S_pattern≈0.99899, S_total≈0.99899`
- 行 2436–2469：
  - `row_index=67, CountyName,mxrshxll, S_total≈0.9988`
  - `row_index=69..95, CountyName,lauderdale / laudxrdalx / laudexdale ...`
  - `row_index=119..126, CountyName,crenshaw ...`
  - 所有这些 S_total 也是 ≈0.9988

可以看到：

- `lauderdale`（正确值）、`laudxrdalx`（错拼）、`de kalb`（正确）、`dx kalb`（错拼）  
  统统被 assign 了几乎一样的高分，接近 1.0。
- threshold=0.5 时，它们全部被判为错误。

这完全体现了上面的推理：  
**CountyName 这一列的 EM 根本没有分辨“正确 vs 错误”，它只是把“不在 top10 的县名”整体当成高风险。**

于是：

- ground truth 上有 60 个 CountyName 错误
- 模型预测了 513 个错误
  - 其中 60 个是真阳性（TP）
  - 453 个是假阳性（FP）
- ⇒ precision ≈ 60 / (60+453) = 0.1169  
  这和 log 里的评估结果完全一致：

  > CountyName: precision=0.1169, recall=1.0, true_positives=60, false_positives=453, ...

---

**7. 总结成一句话：性能差的核心原因**

综合上面：

- 规则层面：
  - CountyName 的 clean 规则里有多条“top10 白名单”规则（accuracy_2, pattern_9）
  - 在 seeds + synthetic dirty 的有限视角里，它们看起来非常“可信”（α 高、β 低）
  - 这些规则在 EM 里变成“只要不是 top10，就极大倾向于脏”的强信号
- 数据分布层面：
  - 真实数据里，正确的县名远远多于 top10，那些低频正确县全部被这些规则打成“可疑”
- 算法层面：
  - clean seed 选择偏向高频县，进一步加剧了“top10=正常、非 top10=异常”的偏见
  - synthetic dirty 只在高频县附近造样本，不能纠正这种偏见
  - EM 在这样的 α/β + prior 下，为了解释观测到的大量“非 top10 行”，只好把它们大部分标成 y=1（脏）

所以，CountyName 上现在的极低 precision 不是简单的 bug，而是：

> **频率驱动的“白名单规则”+ 偏置的 clean seed + 不充分的 dirty 样本**  
> 把“稀有但正确的县名”整体当成了错误，导致 EM 高估了这一列的错误比例。

如果你之后想针对 CountyName/高基数类别专门做改进，可以进一步讨论几个方向，比如：

- 降低或禁用“topN 白名单”类规则在这类列上的权重
- 让 synthetic dirty 明确生成“非 top10 但正常”的样本，打破这种频率偏见
- 或者在 seed 选择阶段，强制覆盖更多不同县名，避免只用高频县做 clean seeds

但在当前这次 run 里，性能差的根因就是上面这条“top10 白名单 + EM 推断”链条。


# Welkinni

2 那从框架角度来看，目前的问题是什么呢？

# Builder with MCP

从“框架”的角度看，现在暴露出来的问题可以分成四层：**规则层 → 监督构造层（seed + synthetic dirty）→ EM 标签模型层 → 决策层**。CountyName 只是把这些问题集中放大了出来。

---

**1. 规则层：把“频率信息”硬编码成了强 clean 规则**

- 现在的 categorical 规则里有一类典型形式：  
  “取 top10 高频值，构造 `value in [top10]` 这样的 clean 规则”，CountyName 就是：
  - `accuracy_2`: `value.lower() in [jefferson, etowah, ..., calhoun]`
  - `pattern_9`: 同样的 top10 白名单
- 在 clean_em 框架里，所有规则都被当成**“潜在可靠的 clean 规则”**，然后通过 seeds + synthetic dirty + EM 学 α / β。
- 对高基数、长尾分布的列（CountyName 典型）来说，这类 “topN 白名单” 实际上等价于一种强烈的**闭世界假设**：
  - “只要不是 top10 就可疑”
- 一旦这类规则进入 EM，而且在 seeds / dirty 上表现很好，它们会变成非常强的统计信号，后续所有层都绕不过去。

框架层面的问题是：  
**没有对这类“频率驱动的白名单规则”做任何正则或结构约束，它们和普通 pattern/长度规则享受同样的地位。**

---

**2. 监督构造层：clean seeds 和 synthetic dirty 都是有偏的**

这一层是现在框架的核心创新点，但也同时是 CountyName 出问题的根源：

- **clean seeds 的偏置**
  - 选 seed 的逻辑是：用全部 clean 规则打分，然后取 top p%（比如 10%）作为高置信 clean。
  - 对 CountyName 这种典型 Zipf 分布的列，得分最高的几乎必然都是：
    - 高频医院 + 高频县（jefferson, houston, marshall, …）
  - 于是：  
    **L_clean 只覆盖了“高频县”，几乎不包含低频但正确的县名。**
- **synthetic dirty 的偏置**
  - DirtyExampleAgent 也是从 seeds 抽值+元数据来造脏数据：
    - CountyName 的脏例子主要是 `houston`、`coffee` 等高频县的变体（houstn, cofee, houston123, …）
  - 它没有刻意生成“正常、但不在 top10 的县名”，也没有生成“低频县的错拼写与混淆”。
  - 结果：L_dirty 也**几乎只在 top10 附近活动**。

所以对“top10 白名单”规则来说，三块数据是这样的：

- clean seeds：几乎全是 top10 → 规则上全 1
- synthetic dirty：只在 top10 上搞错 → 规则上大部分是 0
- 未标注真实数据 U：大量低频县名 → 规则上大量 0

在这个视角下，“属于 top10” 看起来就是一个**完美区分 clean/dirty 的强特征**，而“低频但正确”的那些县根本没出现在有标签的两块数据里。

框架层面的问题是：  
**我们用“规则自己 + 它们选出来的 seeds”来构造监督，没有任何机制去保证 seed / dirty 的覆盖度和多样性，导致监督本身在高基数列上极度偏置。**

---

**3. EM 标签模型层：完全信任有偏监督，缺乏全局约束**

在 `_run_em_for_family` 里（单 family 的 EM）：

- 初始：
  - α_r 用 clean_z 的均值估 → “通过率在 clean seed 上有多高”
  - β_r 用 dirty_z 的均值估 → “在 synthetic dirty 上漏报率有多低”
- 对 CountyName 的 top10 白名单规则：
  - α ≈ 1（所有 seed 都在 top10）  
  - β ≈ 0（脏例大多是错拼、带数字的变体，不在白名单）
- EM 在迭代过程中会把这些“极端的 α/β”固定下来，然后在未标注数据 U 上做推断：

  - 对不在 top10 的真实行，白名单规则给 `z=0`
  - 在当前参数下：
    - P(z=0 | y=0) 很小（因为 α≈1）
    - P(z=0 | y=1) 很大（因为 β≈0）
  - 于是这条规则强力推动“y=1（脏）”。

再叠加：

- prior_dirty 是 per-family 固定的（比如 0.05），但没有对“每列的总体错误率”施加额外约束；
- EM 更新 π（脏先验）时，只基于当前解释最优，没有 global regularization。

在 CountyName 上，为了“解释”大量非 top10 的行，最优解就是：

- 把 π（这一 family 下的脏比例）推高
- 让 gamma 对大部分非 top10 行都接近 1

这就是 log 里看到的：

- gamma_mean≈0.58，513/900 行 gamma>0.5  
- 最后 513 个 cell 被判为 error，真实只有 60 个 error。

框架层面的几个问题：

- EM 模型**没有显式建模 “规则本身可能是有偏/错误的”**；  
  它假定 seeds + synthetic dirty 给出的 α/β 是可信的，并据此解释所有 U。
- 没有任何 per-column 的硬约束或 soft prior，比如：
  - “这一列的脏比例不太可能从 5% 飙到 60%”
  - “如果某个 family 的 gamma_mean 非常高，但 β 又极端地低，是否说明 supervision/规则有问题？”
- 不同 family 是完全独立跑 EM 的，后面再用 max 合并，这进一步放大了任何一个 family 的极端判断。

---

**4. 决策层：S_total=max(...) + 全局统一阈值，加剧了偏差**

最后一层的逻辑（[main.py:620-637](file:///mnt/data/welkinni/table_det/main.py#L620-L637)）：

- 对每个 cell，取：
  - `S_total = max(S_missing, S_outlier, S_pattern)`
- 再统一用一个 `score_threshold`（例如 0.5）做二值化。

对 CountyName 来说：

- outlier / pattern 的 gamma 都在 0.98–0.999 这个量级，missing family 被 drop 掉不参与。
- 所以只要这两家中的任何一个给高分（事实上两家都给了），`S_total` 就 ≈ 0.999。
- 阈值 0.5 在这里完全形同虚设。

如果框架想在没有真标签的情况下“学出软分布”而不是“全局一刀切”，类似的 max + 固定阈值策略其实有几个问题：

- **max 聚合**：只要有一个 family 偏到极端，整列就被它“绑架”了；
- **阈值固定**：没有 per-column / per-family 自适应的决策机制，所有列共享 0.5；
- **事后评估虽然看到了低 precision，但并没有反馈回模型**——框架是单向的：EM→打分→评估，不会根据评估去调约束。

---

**5. 抽象一下：框架层面的“结构性问题”**

把上面的细节抽象掉，其实是三条核心结构问题：

1. **监督构造高度自举化，却没有抗偏机制**
   - 用规则本身选 seeds，再用 seeds + synthetic dirty 去估规则质量；
   - 这一套在中等难度的列上没问题，但在“高基数 + 长尾分布”的列上，非常容易坍塌成：
     - “我只见过 top10，是好人”；  
     - “我也只造了 top10 周边的坏人”；  
     - 然后 EM 就会在这个小圈子里自嗨，把圈外的所有人都当坏人。

2. **标签模型对规则的信任太高，对“列级错误率”的先验太弱**
   - α/β 一旦被 seeds/dirty 决定，就几乎没机制把“这组 α/β 本身是错的”反馈出来；
   - EM 可以自由把 π 推到很高，只要能解释观测数据；
   - 没有列级/数据级的 regularization 去压制“整列 60% 都是错”的极端结论。

3. **最终决策把任何一个 family 的极端判断放大成“整列灾难”**
   - max 聚合 + 固定阈值，使得某个 family 一旦出问题，整个列的 S_total 都非常大；
   - 缺乏：
     - per-column adaptive threshold（根据 gamma 分布来调）；
     - 或者基于秩/比例的筛选（比如每列最多报前 p% 作为 error）。

CountyName 的现象（513 个候选 + precision≈0.12）只是一个实例；  
**任何“高基数、长尾、且规则里有 topN 白名单”的 categorical 列，现在在 clean_em 框架下都可能重演这件事。**

---

如果你接下来想从框架上动刀，我会从这三个方向设计改动：

- seed/dirty 这一层：打破 “高频=clean 全部” 的假设，引入覆盖度和多样性约束；
- EM 层：给 per-column 的错误率加上合理的先验或约束，避免 π 被推到极端；
- 决策层：摒弃“max+统一阈值”，换成 per-column 自适应的决策策略（比如基于分位数或 mixture 拟合）。

之后我们可以挑一条先落地在 CountyName 上做一个对照实验。
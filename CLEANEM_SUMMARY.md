**`clean_em`，本质上是“弱监督证据融合 + 少量主动标注校准”的单元格打分器。**  
名字里虽然有 `EM`，但当前主路径核心其实是：`metadata/rules -> evidence -> feature matrix -> calibrated posterior`，入口在 [main.py](/data/nw/table_det/main.py:2301)。

**算法流程**

1. **读入脏表，做 profiling**  
输入：`dirty_csv`  
输出：`df` 和每列 `metadata`  
内容包括列类型、缺失 token、regex 候选、shape 分布、prototype、relationship profile 等，见 [profiler.py](/data/nw/table_det/profiler.py:75)。

2. **按列生成 clean rules**  
输入：`metadata`  
输出：`rule_pool[column] = [rule1, rule2, ...]`  
规则被映射到 4 个 family：
`completeness -> missing`，`accuracy -> outlier`，`pattern_consistency -> pattern`，`column_relationship -> relationship`，见 [main.py](/data/nw/table_det/main.py:220)。

3. **建立两层对象**
- `value_registry`：同一列里相同 normalized value 聚成一组  
- `cell_registry`：每个 cell 一个对象  
见 [main.py](/data/nw/table_det/main.py:1649)。  
这一步的关键是把“值级别重复”与“单元格级别上下文”分开建模。

4. **抽取证据 evidence**  
输入：`df + metadata + rule_pool + registries`  
输出：
- `value_observations`
- `cell_observations`  
见 [main.py](/data/nw/table_det/main.py:1698)。  
典型 evidence 有：
- value 级：`missing_token`、`parse_failure`、`regex_pass`、`rarity_high`、`prototype_close/far`
- cell 级：`pattern_match/mismatch`、`relationship_satisfied/violation`、`contextual_agreement/disagreement`

5. **从 evidence 汇总 hard labels**  
输入：observations  
输出：`hard_labels["value"]` 和 `hard_labels["cell"]`  
见 [main.py](/data/nw/table_det/main.py:2144)。  
逻辑很直接：只要某个 target 上出现了 `hard=True` 的 dirty evidence，就标成 hard dirty。

6. **把 evidence 变成 cell-feature 矩阵**  
输入：observations  
输出：`EvidenceMatrix`  
见 [cleanem_inference.py](/data/nw/table_det/cleanem_inference.py:68)。  
这里每个 cell 是一行，每种 evidence source 是一列：
- dirty evidence 记正值
- clean evidence 记负值

7. **主动选择少量 query 做校准**  
输入：`EvidenceMatrix + budget`  
输出：`selected_cell_keys`  
见 [cleanem_inference.py](/data/nw/table_det/cleanem_inference.py:117)。  
它不是挑“最不确定”的 cell，而是挑“最能帮助学清楚证据权重”的代表性样本。

8. **用少量标注学证据权重，给所有 cell 出 posterior**  
输入：`EvidenceMatrix + selected labels + base prior`  
输出：
- `cell_posteriors`
- `value_priors`
- `calibration_trace`  
见 [cleanem_inference.py](/data/nw/table_det/cleanem_inference.py:218)。  
最后每个 cell 会得到一个 `cell_posterior`，表示“这个 cell 是 dirty 的概率/分数”。

9. **导出结果并按阈值判错**  
输出文件是 `*_clean_em_scores.csv`，每行包含：
`row_index, column, value, value_prior, cell_posterior, hard_label, evidence_summary...`  
然后用 `cell_posterior >= score_threshold` 作为最终检测结果。

**一个从头到尾的例子**

拿 `hospital` 里一个典型脏值：`row=120, column=Score, value='x00%'`。

1. **输入**  
脏表里这一格是 `'x00%'`。

2. **profiling**  
`Score` 列会被识别成比较稳定的 numeric/percent 型列，所以它有较强 numeric contract。

3. **rule 生成**  
这一列通常会有“像百分比”“能解析成数值”“范围合理”之类的 clean rule。

4. **registry**
- `value_key = "Score::x00%"`
- `cell_key = "120::Score"`

5. **evidence 抽取**
- value 级：
  - `parse_failure: dirty`
  - `type_conflict: dirty`
- cell 级：
  - `cell_parse_failure: dirty`
  - `cell_type_conflict: dirty`
  - 可能还有一点 `contextual_disagreement`

因为这列 numeric contract 很强，这几个 parse/type 失败会被记成 **hard dirty**。

6. **hard label**
- `hard_label(cell) = dirty`

7. **active query**
如果这一格被选中，就去 `clean_csv` 看对应位置。  
假设 clean 值是 `100%`，那这格的 oracle label 就是 `dirty=1`。

8. **posterior**
在 evidence matrix 里，这一格对应的 feature row 会有一串正的 dirty 证据。  
校准后它通常会得到一个比较高的 `cell_posterior`，比如日志里大概 `0.82` 左右。

9. **最终输出**
在结果 CSV 里，这一行会像：
- `value='x00%'`
- `value_prior≈0.81`
- `cell_posterior≈0.82`
- `hard_label='dirty'`

因为超过阈值 `0.5`，最终被判成错误。

**这套方案的核心思想**

它不是直接问“这条规则过不过”，而是问：

- 这个值本身像不像脏值？
- 这个 cell 在本行本列上下文里像不像脏值？
- 这些证据在当前数据集上到底靠不靠谱？
- 用少量标注后，应该把哪些证据看重一些？

所以最后输出的不是一个硬规则判断，而是一个**融合了 value 级、cell 级、上下文级证据的 posterior 分数**。

如果你想，我下一条我可以把它再压成一张更直观的“对象流转图”，比如 `dirty_csv -> metadata -> rule_pool -> observations -> evidence_matrix -> posteriors` 这种。



**可解释性**

## 1. 现在已经有的解释性

### A. 证据级解释
导出的结果里已经保留了：
- `hard_label`
- `hard_reason_summary`
- `value_evidence_summary`
- `cell_evidence_summary`

见 [main.py](/data/nw/table_det/main.py:2260)。

这意味着对每个被判错的 cell，你已经能回答：

- 它有没有被 hard dirty 直接锚定？
- 值级别有哪些证据支持它脏/干净？
- cell 级别有哪些证据支持它脏/干净？

比如一条结果可能会长成：
```text
hard_label=dirty
hard_reason_summary=numeric_parse_failure;numeric_type_conflict
value_evidence_summary=parse_failure:dirty:1.00:hard;type_conflict:dirty:1.00:hard
cell_evidence_summary=cell_parse_failure:dirty:1.00:hard;cell_type_conflict:dirty:1.00:hard
```

这已经能解释：
> “为什么它被打高分？”  
> 因为它违反了强 numeric contract，而且是硬证据。

### B. 模型级解释
日志里会输出：
- `learned_evidence_weights`
- `Active query`
- `value_evidence_source_coverage`
- `cell_evidence_source_coverage`

见 [main.py](/data/nw/table_det/main.py:2461)

这能解释：
- 这次校准到底学会了“更信哪些证据”
- 主动标注看了哪些样本
- 哪些 evidence source 覆盖范围大

所以模型层你也能回答：

> “为什么这次整体偏向抓 missing / pattern mismatch / contextual disagreement？”

因为 learned weights 里这些 source 权重大。
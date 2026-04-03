# 55题共同集合 Strict 对比归档

## 背景

本报告用于归档三组实验结果在统一口径下的 strict 分数对比：

- `runs_gemini_3_pro`
- `runs_qwen_3_4B`
- `runs_qwen_3.5_plus`

此前仓库中的 `rule_dimension_scores` 只覆盖了少量已在 `data/rule_sets/implicit_rules_zh.yaml` 中显式挂标签的场景，不能代表全部题目。为得到更完整的维度对比，本次在保留原有 14 道已标注场景的基础上，对其余 56 道未标注场景依据 `implicit_reason`、`hidden_constraints`、`suggested_plan`、`execution_rules`、`evaluation_rubric` 做了规则家族补充归类，再统一计算 strict 分数。

## 统计口径

### 共同 55 题集合

三组结果原始总题数均为 70。为避免 `runs_gemini_3_pro` 中配额失败造成不公平比较，本次统一剔除 Gemini 中明确属于 `429 RESOURCE_EXHAUSTED` 的 15 道题，保留其余 55 道题作为共同对比集合。

被剔除的 15 道题为：

- `cancel_may26_houston_flight`
- `cancel_mistaken_pending_order`
- `cancel_orlando_trip_reservation`
- `comedy_movie_night_options`
- `check_before_canceling_mistaken_order`
- `cancel_pending_action_camera_order`
- `cancel_denver_miami_flight_tomorrow`
- `review_mistaken_order_before_cancel`
- `confirm_accidental_order_before_cancellation`
- `check_laptop_order_before_shipping`
- `perseverance_movies_for_middle_schoolers`
- `family_friendly_animated_movie_under_90`
- `chicago_picnic_weather_window`
- `review_sf_round_trip_safely`
- `cancel_just_placed_order_safely`

### Strict 计分方式

每道题的 `strict_scenario_score` 直接取自 batch summary 中的逐题结果：

- 全部检查项通过：记 `1.0`
- 仅差 1 项：记 `0.5`
- 差 2 项及以上：记 `0.0`

统一到 55 题集合后：

- 若某模型该题完成，则取该题已有 `strict_scenario_score`
- 若某模型该题失败，则该题 strict 按 `0` 计入

### 规则维度统计方式

- 每道题可同时属于多个规则维度
- 某规则维度的 `total` 为该维度下所有题目的 strict 分数求和
- 某规则维度的 `avg` 为该维度 `total / 该维度题数`
- 由于是多标签统计，各维度题数之和会大于 55

## 总体 Strict 对比

| 模型 | Strict 总分 | Strict 均分 |
|---|---:|---:|
| `runs_qwen_3.5_plus` | `39.5` | `0.718` |
| `runs_qwen_3_4B` | `36.0` | `0.655` |
| `runs_gemini_3_pro` | `35.0` | `0.636` |

### 总体结论

按共同 55 题的 strict 表现排序：

1. `runs_qwen_3.5_plus`
2. `runs_qwen_3_4B`
3. `runs_gemini_3_pro`

`runs_qwen_3.5_plus` 在统一口径下仍然最强；`runs_qwen_3_4B` 与 `runs_gemini_3_pro` 接近，但略占优势。

## 规则维度 Strict 对比

| 规则维度 | 55题中标注数 | Gemini total | Gemini avg | Qwen 3 4B total | Qwen 3 4B avg | Qwen 3.5+ total | Qwen 3.5+ avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `deadline_buffer` | 7 | 5.5 | 0.786 | 3.0 | 0.429 | 6.0 | 0.857 |
| `conflict_and_exclusivity` | 6 | 4.5 | 0.750 | 5.5 | 0.917 | 4.0 | 0.667 |
| `safety_and_irreversibility` | 24 | 19.5 | 0.813 | 18.0 | 0.750 | 20.5 | 0.854 |
| `latent_goal_vs_literal_request` | 26 | 17.0 | 0.654 | 17.5 | 0.673 | 18.5 | 0.712 |
| `multi_hop_evidence` | 31 | 22.0 | 0.710 | 22.5 | 0.726 | 24.0 | 0.774 |
| `side_effect_optimization` | 7 | 4.5 | 0.643 | 4.0 | 0.571 | 4.5 | 0.643 |
| `explanation_obligation` | 22 | 14.5 | 0.659 | 13.0 | 0.591 | 15.5 | 0.705 |

## 分维度观察

### `deadline_buffer`

- 最强：`runs_qwen_3.5_plus`
- 次强：`runs_gemini_3_pro`
- `runs_qwen_3_4B` 在此维度落后明显

### `conflict_and_exclusivity`

- 最强：`runs_qwen_3_4B`
- 其次：`runs_gemini_3_pro`
- `runs_qwen_3.5_plus` 在该维度相对偏弱

### `safety_and_irreversibility`

- 最强：`runs_qwen_3.5_plus`
- 其次：`runs_gemini_3_pro`
- 三者整体都较强，但 `runs_qwen_3.5_plus` 最稳定

### `latent_goal_vs_literal_request`

- 最强：`runs_qwen_3.5_plus`
- `runs_qwen_3_4B` 与 `runs_gemini_3_pro` 接近

### `multi_hop_evidence`

- 最强：`runs_qwen_3.5_plus`
- 次强：`runs_qwen_3_4B`
- `runs_gemini_3_pro` 略低

### `side_effect_optimization`

- `runs_qwen_3.5_plus` 与 `runs_gemini_3_pro` 持平
- `runs_qwen_3_4B` 略低

### `explanation_obligation`

- 最强：`runs_qwen_3.5_plus`
- 次强：`runs_gemini_3_pro`
- `runs_qwen_3_4B` 相对较弱

## 结果解读

### 为什么这次维度结果比之前更可信

此前仓库里的 `rule_dimension_scores` 是通过 `data/rule_sets/implicit_rules_zh.yaml` 中的 `existing_scenarios` 生成的，只覆盖了少数已显式挂标签的场景，因此会产生以下偏差：

- 很多题虽然有 `execution_rules`，但 `summary.rule_ids` 为空
- 维度分只基于少量样本，容易出现某模型“各项都很高”的假象
- 失败题如果未进入该维度映射，就不会拉低该维度得分

本次补齐 56 道未标注题的规则归类后，维度统计覆盖了共同 55 题中全部场景，因此更适合用来做模型之间的横向比较。

### 这次最值得记住的结论

- 如果只看共同 55 题 strict 总体表现，`runs_qwen_3.5_plus` 最好
- `runs_qwen_3_4B` 的总体 strict 略强于 `runs_gemini_3_pro`
- `runs_qwen_3_4B` 在冲突检测类题上表现最好
- `runs_qwen_3.5_plus` 在时间缓冲、安全性、多跳证据、真实目标识别、解释义务等大多数维度上更强
- `runs_gemini_3_pro` 并非“所有规则都更强”，之前主要是维度覆盖不足带来的观感偏差

## 备注

- 本文中的 56 道补充规则标签为基于场景内容的人工归纳结果，适合分析与归档，但不等同于仓库中已正式写入的官方标签。
- 若后续要长期复用这份维度统计，建议把这些补充标签正式回填到规则映射配置中，避免后续批跑仍出现 `rule_ids: []` 的情况。

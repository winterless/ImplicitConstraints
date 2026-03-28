# Toucan Query Generation Guide

这个目录放的是从 `Toucan-1.5M-resolve` 里抽取高质量 query 候选的两步脚本，以及一个可直接维护的 regex 配置文件。

当前默认数据源：

- `/home/unlimitediw/workspace/ImplicitConstraints/seed_datasets/Toucan-1.5M-resolve`

说明：

- 这个路径当前是一个软链接，指向你之前整理出来的 `Toucan-1.5M-resolve`
- 所以在 `ImplicitConstraints` 工程里可以直接把它当作新数据源使用

## Files

- `export_resolve_quality_csv.py`
  - 从 resolved jsonl 里抽取筛选所需字段，导出为一个中间 `csv`
- `filter_resolve_quality_topk.py`
  - 读取中间 `csv`，按 regex、长度、质量分数进行过滤和排序，输出 top-k
- `filter_resolve_quality_patterns.txt`
  - 一行一个 regex，用来做前置白名单过滤

## Workflow

这套流程分成两步：

1. 原始 jsonl -> 中间质量表
2. 中间质量表 -> 最终 top-k 结果

拆成两步的原因是后续你通常会频繁调整：

- regex 关键词
- 长度阈值
- top-k
- 打分权重

如果保留中间 `csv`，就不需要每次都重新扫描整个 Toucan 数据目录。

## Step 1

在当前目录运行：

```bash
cd /home/unlimitediw/workspace/ImplicitConstraints/src/query_gen
python export_resolve_quality_csv.py
```

默认输入：

- `/home/unlimitediw/workspace/ImplicitConstraints/seed_datasets/Toucan-1.5M-resolve`

默认输出：

- `/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality.csv`

导出的字段：

- `source_file`
- `line_number`
- `uuid`
- `content`
- `first_user_content`
- `question_quality_avg`
- `response_quality_avg`

当前评分定义：

- `question_quality_avg`
  - 取 `question_quality_assessment.question_quality.score`
  - 和 `question_quality_assessment.scenario_realism.score`
  - 的平均值
- `response_quality_avg`
  - 取 `response_quality_assessment.completeness.score`

跳过规则：

- 如果上述任一分数为空，跳过
- 如果上述任一分数为 `0`，跳过

## Step 2

在当前目录运行：

```bash
cd /home/unlimitediw/workspace/ImplicitConstraints/src/query_gen
python filter_resolve_quality_topk.py
```

默认输入：

- `/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality.csv`

默认输出：

- `/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality_top500.csv`

当前默认规则：

- 先用 `filter_resolve_quality_patterns.txt` 做 regex 白名单过滤
- 只有命中 regex 的 `content` 才进入后续流程
- `content` 长度必须小于 `150`
- 排序分数为：

```text
rank_score = q_weight * question_quality_avg - r_weight * response_quality_avg
```

默认参数：

- `top_k = 500`
- `max_chars = 150`
- `q_weight = 1.0`
- `r_weight = 1.0`

## Pattern File

默认 pattern 文件：

- `/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/filter_resolve_quality_patterns.txt`

维护方式：

- 一行一个 regex
- 空行会跳过
- 以 `#` 开头的行会被当成注释跳过

例如：

```text
来得及
能不能赶上
最晚.*出发
reason
how should i choose
```

修改完这个文件后，只需要重新执行第二步：

```bash
python filter_resolve_quality_topk.py
```

## Useful Commands

只重跑第一步：

```bash
python export_resolve_quality_csv.py
```

只重跑第二步：

```bash
python filter_resolve_quality_topk.py
```

自定义输出文件：

```bash
python src/query_gen/export_resolve_quality_csv.py -o /home/unlimitediw/workspace/ImplicitConstraints/tmp/toucan_quality.csv
python src/query_gen/filter_resolve_quality_topk.py -i /home/unlimitediw/workspace/ImplicitConstraints/tmp/toucan_quality.csv -o /home/unlimitediw/workspace/ImplicitConstraints/tmp/toucan_top500.csv
```

自定义 top-k 和长度阈值：

```bash
python filter_resolve_quality_topk.py --top-k 1000 --max-chars 200
```

使用别的 pattern 文件：

```bash
python filter_resolve_quality_topk.py --pattern-file /path/to/patterns.txt
```

## Suggested Editing Points

如果你后续要继续在这个工程里扩展，通常只需要改这几个地方：

- 改 regex 白名单：`filter_resolve_quality_patterns.txt`
- 改长度或 top-k：`filter_resolve_quality_topk.py`
- 改评分口径：`export_resolve_quality_csv.py`

## Quick Start

如果只是想从头到尾跑一遍：

```bash
cd /home/unlimitediw/workspace/ImplicitConstraints/src/query_gen
python export_resolve_quality_csv.py
python filter_resolve_quality_topk.py
```

最终结果在：

- `/home/unlimitediw/workspace/ImplicitConstraints/src/query_gen/toucan_resolve_quality_top500.csv`


python src/query_gen/export_resolve_quality_csv.py

python src/query_gen/filter_resolve_quality_topk.py   -i src/query_gen/seed_dataset_query_stats.csv   --top-k 100   --max-query-chars 80   --sort-field turn_count   -o src/query_gen/seed_dataset_turn_a.jsonl
# Implicit Constraints Demo

一个最小可运行的 `Agent-as-a-World` demo，实现了：

- MCP 风格的 mock 工具接口
- 单场景 YAML 加载
- 可插拔 agent
- 可切换的 world model（deterministic mock / LLM world model）
- criterion-level evaluator
- 一个 `route + time` 的完整 demo

## Demo 内容

当前内置场景：`airport_route_time`

用户请求：

`帮我设置导航，确保我能准时到机场。`

正确 agent 需要：

- 先查询日历里的下一个事件
- 推断航班需要提前到达
- 再查询路线
- 选择真正能满足时间约束的路线

## 安装

```bash
python -m pip install -e .
```

## 运行

```bash
python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml \
  --world-mode mock
```

如果你想强制使用本地 deterministic world：

```bash
python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml \
  --world-mode mock
```

如果已经配置了 API key，并希望更贴近论文里的 Agent-as-a-World 设计，可以让 world 也由单独的 LLM 来模拟：

```bash
python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml \
  --world-mode llm
```

默认会把完整运行结果保存到：

```text
runs/airport_route_time.json
```

如果想自定义输出目录：

```bash
python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml \
  --world-mode mock \
  --output runs/custom
```

这会输出到：

```text
runs/custom/airport_route_time.json
```

如果不传 `--scenario`，会自动扫描 `data/scenarios/` 下所有可执行场景并批量运行，跳过像 catalog 这类不可执行 YAML：

```bash
python -m implicit_constraints_demo.main \
  --world-mode mock
```

默认批量结果会输出到：

```text
runs/batch/
```

其中每个场景一个结果文件，另外还会生成：

```text
runs/batch/_summary.json
```

## 模型 API

当前 agent 默认按 `DataBot` 里的调用方式接 `DashScope` 兼容接口：

- `base_url`: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `model`: `qwen-plus`

API key 读取顺序：

1. `--api-key`
2. 环境变量 `DASHSCOPE_API_KEY`
3. 环境变量 `ALIYUN_API_KEY`
4. `.secrets/alicloud_api_key.txt`

`.secrets/` 已加入 `.gitignore`，避免误提交密钥。

如果没有配置 API key，可以配合 `--world-mode mock` 使用；若选择 `--world-mode llm`，CLI 会直接报错而不是自动回退。`--world-mode` 现在必须显式传入。

## 目录结构

```text
data/
  scenarios/
  tool_schemas/
src/implicit_constraints_demo/
  agent.py
  evaluator.py
  main.py
  orchestrator.py
  schemas.py
  tool_registry.py
  world.py
```

## 设计原则

- 先做少量高质量 demo，不追求大而全
- 工具接口优先做成 MCP 风格，方便后续替换成真实 server
- evaluator 先做 deterministic checks，避免过早引入额外 LLM 漂移
- agent 保持可插拔，目前内置一个基于 Qwen API 的 planning agent

## 下一步可扩展方向

- 增加 `alarm`、`calendar CRUD`、`weather`、`maps` 的更多 mock 工具
- 增加第二个 agent 实现，例如本地模型或其他 provider adapter
- 增加更多 scenario
- 把本地 mock registry 升级成真正的 MCP server

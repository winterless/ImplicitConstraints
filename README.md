# Implicit Constraints Demo

一个最小可运行的 `Agent-as-a-World` demo，实现了：

- MCP 风格的 mock 工具接口
- 单场景 YAML 加载
- 可插拔 agent
- 可分别配置的 `agent / world / evaluator` 三个角色
- 可切换的 world model（deterministic mock / LLM world model）
- evaluator 支持 deterministic checks 或 LLM judge
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

默认会读取根目录下的 `llm_config.yaml`。如果不额外传参数，`agent / world / evaluator` 三个角色都会按配置文件运行；仓库默认配置是三者都走 LLM。

这个仓库使用的是 `src/` 布局，所以有两种运行方式：

1. 先安装包，再直接运行：

```bash
python -m pip install -e .
python -m implicit_constraints_demo.main --scenario data/scenarios/airport_route_time.yaml
```

2. 不安装包，直接从仓库根目录运行：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml
```

如果你已经在配置文件里填好了可用的 LLM endpoint / API key，直接运行：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml
```

如果你想强制使用本地 deterministic world：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml \
  --world-mode mock
```

如果已经配置了 API key，并希望更贴近论文里的 Agent-as-a-World 设计，可以让 world 也由单独的 LLM 来模拟：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --scenario data/scenarios/airport_route_time.yaml \
  --world-mode llm
```

默认会把完整运行结果保存到：

```text
runs/airport_route_time.json
```

如果想自定义输出目录：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
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
PYTHONPATH=src python -m implicit_constraints_demo.main \
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

## 模型配置

默认配置文件：`llm_config.yaml`

```yaml
agent:
  mode: llm
  provider: openai_compatible
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen-plus
  api_key_env: DASHSCOPE_API_KEY,ALIYUN_API_KEY
  api_key_file: .secrets/alicloud_api_key.txt
  require_api_key: true
```

三个角色都可以独立配置：

- `agent`: 负责推理和选工具
- `world`: 负责模拟 MCP / 工具返回值
- `evaluator`: 负责给最终运行结果打分

`mode` 支持：

- `agent`: `llm | heuristic`
- `world`: `llm | mock`
- `evaluator`: `llm | deterministic`

当前默认按 `DashScope` 兼容接口配置：

- `base_url`: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `model`: `qwen-plus`

如果要切到本地 OpenAI-compatible 服务，可以直接改配置文件，例如：

```yaml
world:
  mode: llm
  provider: local_openai_compatible
  base_url: http://127.0.0.1:8000/v1
  model: qwen2.5-7b-instruct
  api_key_env: ""
  api_key_file: ""
  require_api_key: false
```

API key 读取顺序：

1. `--api-key`
2. 该角色配置里的 `api_key_env`
3. 该角色配置里的 `api_key_file`

`.secrets/` 已加入 `.gitignore`，避免误提交密钥。

按不同配置文件运行的常用命令：

1. 使用默认配置 `llm_config.yaml`：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --config llm_config.yaml \
  --scenario data/scenarios/airport_route_time.yaml
```

2. 显式使用 `llm_config_qwen3.5plus.yaml`：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --config llm_config_qwen3.5plus.yaml \
  --scenario data/scenarios/airport_route_time.yaml
```

3. 使用 `llm_config_qwen3local.yaml`，让 `agent` 走本地 OpenAI-compatible 服务：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main \
  --config llm_config_qwen3local.yaml \
  --scenario data/scenarios/airport_route_time.yaml
```

其中 `llm_config_qwen3local.yaml` 里的 `agent.base_url` 指向 `http://127.0.0.1:8000/v1`，所以需要先启动本地兼容 OpenAI 的推理服务；同时该配置里的 `world` 和 `evaluator` 仍然使用 DashScope，因此对应 API key 也仍需可用。

如果想批量跑 `data/scenarios/` 下的所有场景，也可以直接替换配置文件：

```bash
PYTHONPATH=src python -m implicit_constraints_demo.main --config llm_config.yaml
PYTHONPATH=src python -m implicit_constraints_demo.main --config llm_config_qwen3.5plus.yaml
PYTHONPATH=src python -m implicit_constraints_demo.main --config llm_config_qwen3local.yaml
```

常用覆盖参数：

- `--config llm_config.yaml`
- `--agent-mode heuristic`
- `--world-mode mock`
- `--evaluator-mode deterministic`
- `--base-url ...`
- `--model ...`
- `--allow-missing-api-key`

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
- evaluator 同时支持 deterministic checks 和 LLM judge，前者更稳，后者更适合 rubric 尚未补全的场景
- agent 保持可插拔，目前内置一个基于 Qwen API 的 planning agent

## 下一步可扩展方向

- 增加 `alarm`、`calendar CRUD`、`weather`、`maps` 的更多 mock 工具
- 增加第二个 agent 实现，例如本地模型或其他 provider adapter
- 增加更多 scenario
- 把本地 mock registry 升级成真正的 MCP server

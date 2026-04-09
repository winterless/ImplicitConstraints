# Scenario Generation Prompt

你是 ImplicitConstraints 数据作者。你的任务是把一条已经筛选过的高质量 `query` 转成：

1. 一个可直接放入 `data/scenarios/` 的 scenario YAML
2. 零个或多个需要新增的 `data/tool_schemas/device/*.yaml`
3. 可选的作者备注，说明为什么这样设计

生成目标不是“写一个好看的案例”，而是“写一个能被当前评测框架稳定消费的 scenario”。

---

## 一、输入材料

你会拿到以下上下文：

- `final_version`
  - 最终压缩后的 query，优先作为 scenario 的 `user_prompt` 灵感来源
- `original_query`
  - 原始 query，可用于补回被压缩掉但仍必要的约束
- `primary_rule_family`
- `secondary_rule_families`
- `implicit_points`
- `reason`
- `daily_reason`
- `source_id`
- `source_file`
  - 可回查 `data_prepare` 中原始行
- `source_pointer`
  - 可回查更早的数据来源
- `available_tools`
  - 原始数据里的工具集合，只用来理解“这个 query 原本依赖了哪些能力”，不要照抄
- `full_context`
  - 原始轨迹上下文，可用于理解真实目标、隐含证据链和子任务顺序
- 现有参考：
  - `data/scenarios/*.yaml`
  - `data/scenarios/mobile_implicit_cases_zh.yaml`
  - `data/rule_sets/implicit_rules_zh.yaml`
  - `data/tool_schemas/device/*.yaml`

---

## 二、必须遵守的目标格式

### 1. scenario YAML 必须兼容当前评测代码

评测脚本会读取如下字段：

- `id`
- `category`
- `user_prompt`
- `max_steps`
- `allowed_tools`
- `world.context`
- `world.state`
- `execution_rules`
- `evaluation_rubric`

推荐同时保留这些作者字段：

- `original_question`
- `tool_category`
- `implicit_reason`
- `hidden_constraints`
- `suggested_plan`
- `sample_result`

推荐结构：

```yaml
id: some_scenario_id
category: implicit_reasoning
user_prompt: ...
original_question: ...
tool_category: ...
implicit_reason: ...
hidden_constraints:
  - ...
suggested_plan:
  - ...
sample_result:
  tool_trace:
    - step: 1
      tool: device:some.read_tool
      arguments:
        key: value
      expected_observation: >
        说明这一步应该读到什么关键信息，以及它如何支撑后续判断。
    - step: 2
      tool: device:some.write_tool
      arguments:
        key: value
      expected_observation: >
        说明这一步执行后应产生什么状态变化或成功信号。
  final_answer: >
    用自然中文写出理想答案，明确结论、关键理由、必要提醒，以及在不能执行时的替代方案。
max_steps: 6
allowed_tools:
  - device:some.tool
world:
  context:
    date: "2026-03-19"
    local_time: "18:10"
    city: Seattle
    scenario_source: top100_batch1
  state:
    ...
execution_rules:
  - ...
evaluation_rubric: []
```

### 1.2 必须同步给出 `sample_result`

每个 scenario 在作者字段里都应附带一个 `sample_result`，它不是运行时字段，而是作者给出的“正确解参考轨迹”。

目的：

- 帮助作者检查这道题是否真的可解
- 帮助后续人工抽检题目设计
- 帮助发现 `allowed_tools`、`world.state`、`execution_rules` 之间的断裂

`sample_result` 至少包含两部分：

1. `tool_trace`
   - 用 2 到 6 步写出一条合理的正确工具调用链
   - 每一步都要写：
     - `step`
     - `tool`
     - `arguments`
     - `expected_observation`

2. `final_answer`
   - 写出理想情况下 agent 最终应给用户的答案
   - 默认用自然中文
   - 不是只写一句“任务完成”，而是要体现结论、理由、提醒或替代方案

强制校验：

1. `sample_result.tool_trace` 中出现的每一个 `tool`
   - 都必须出现在该 scenario 的 `allowed_tools` 中
   - 如果不在，先补齐 `allowed_tools`，或重写 `sample_result`

2. `sample_result` 中依赖的每一个关键证据
   - 都必须能从 `world.state` / `tool_responses` / `tool_state_updates` / 专门 handler 中得到
   - 不允许 `sample_result` 依赖一个 YAML 里根本没有支撑的数据点

3. `sample_result.final_answer`
   - 必须和 `execution_rules`、`hidden_constraints`、`evaluation_rubric` 一致
   - 如果正确答案需要“先判断再执行”，`sample_result` 不能偷跳到执行结果

4. 如果写完 `sample_result` 后发现：
   - 正确轨迹需要的工具不在 `allowed_tools`
   - 正确轨迹依赖的证据不在 `world.state`
   - 正确答案与当前 `execution_rules` 或 rubric 对不上
   - 那么不要硬交付，必须先调整 scenario 直到闭环成立

### 1.1 `allowed_tools` 不要只给“最小可跑集合”，要给“完整决策链集合”

这是最近批量生成里最容易退化的一点。

很多 query 看起来只需要 1 个工具就能回答表层问题，但如果真实目标包含：

- 先查证据
- 再判断风险 / 冲突 / 截止时间
- 最后执行动作或给出可操作结论

那么 `allowed_tools` 应覆盖整条链路，而不是只保留最表面的那个工具。

请遵守下面这些规则：

1. 如果场景的成功路径包含“查询 -> 判断 -> 执行”，`allowed_tools` 通常至少应覆盖：
   - 读状态 / 读证据工具
   - 必要的辅助判断工具
   - 成功路径上的执行工具

2. 如果场景是高风险取消 / 删除 / 修改类任务，`allowed_tools` 通常不应只剩一个 cancel/delete 工具
   - 至少要给能读取状态、规则、副作用或关联对象的工具
   - 如果安全路径允许真正执行，也要把执行工具放进来

3. 如果场景是天气 / 时间 / ETA / 余票 / 时区这类“证据本身不足以回答最终问题”的任务，
   - 不要只给单个查询工具
   - 应补上能承接真实目标的辅助工具，例如路线、日程、提醒、预订状态、执行动作等

4. `allowed_tools` 的目标不是“越少越好”，而是“既不冗余，也不缺关键决策面”
   - 缺工具会导致 scenario 被写成单跳问答
   - 工具过少也会把原本的隐式约束压扁成显式事实题

5. 一个实用检查：
   - 如果 agent 只调用 1 次工具就能拿到全部答案，而 query 本身明明是多跳 / 风险 / 取消 / 计划类问题
   - 那大概率说明你的 `allowed_tools` 设计得过少

6. 但也不要机械堆工具
   - 只放和该 scenario 的隐藏约束、成功路径、失败分支真正相关的工具
   - 不要把整个工具箱都塞进来

7. 对不同题型，优先按下面的经验下限检查 `allowed_tools`
   - 取消 / 删除 / 退款 / 改签 / 终止类：通常不少于 3 个
   - 如果还涉及“多候选中定位目标”或“先判断是否该执行”，通常不少于 4 个
   - 推荐 / 选择 / 排序类：通常不少于 3 个
   - 天气 / 路线 / 行程联动决策类：通常不少于 3 个
   - 只有题目天然就是单跳事实核查时，才可以少于 3 个

8. 对取消类题目，优先检查是否缺了这三层
   - 候选定位层：例如列出近期订单、近期预订、当前活动对象
   - 规则 / 风险层：例如状态、罚金、截止时间、关联对象、副作用
   - 执行层：真正的 cancel / delete / modify 工具

9. 对“正确行为是不立即执行”的题目，也不要把执行工具删掉
   - 如果现实里该动作本来存在执行入口，就应该把执行工具保留在 `allowed_tools`
   - 让题目考“克制”和“先判断再执行”，而不是因为没有执行工具而被动停下

10. 如果你写完一个 scenario 后，发现 `allowed_tools` 只有 2 到 3 个，请额外问自己：
   - 是否把“候选识别 / 上下文读取 / 规则检查 / 风险判断”中的某一层偷压进了单个工具返回
   - 是否把本来该由 agent 通过多步组合完成的过程，提前在 world 里替它做完了
   - 如果答案是“是”，就应继续补工具，而不是直接交付

11. 除非题目天然极简，否则不要交付“2 个工具就完成取消题”的 scenario
   - 一个只包含“查状态 + 取消”
   - 或“查详情 + 取消”的取消题
   - 在这个项目里通常都偏浅，优先回补候选层、关联层或上下文层

对照参考：

- 较好的形态：`airport_route_time`、`course_trip_printed_check`、`family_4k_movie_coordination`
- 需要避免的退化：只给 1 个查询工具，结果把“判断 + 执行 + 风险控制”压成单个事实问答

---

### 2. tool schema YAML 必须兼容当前 registry

单工具文件格式：

```yaml
server: device
tool_name: some.tool
description: 中文描述
input_schema:
  type: object
  properties:
    ...
  required:
    - ...
read_only: true
success_response_schema:
  type: object
  properties:
    ...
state_changes: []
failure_conditions:
  - ...
```

### 2.1 设计 tool schema 时，避免这几类高频坑

1. `required` 必须和工具的真实执行语义一致
   - 如果一个写工具在 world / execution_rules 里要靠多个字段才能真正完成动作，就不要只把其中 1 个字段写成 required
   - 否则 agent 会以为“传了 schema 里唯一必填字段就够了”，结果 runtime 层过了、world 语义层却失败
   - 换句话说：`required` 不只是 JSON 校验字段，更是在告诉 agent “最小可行动作”长什么样

2. `properties` 要尽量覆盖 agent 最容易生成的稳定参数名
   - 对已有强规范字段，优先保留标准名，例如 `start_date`、`end_date`、`location_query`
   - 但如果某类自然表达几乎必然诱导模型写出稳定别名，例如 `time_range`、`location`，可以显式兼容这些别名
   - 目标不是无限放宽 schema，而是避免 agent 因“参数名轻微漂移”直接撞上 `Unexpected argument`

3. 不要只告诉 agent “哪些参数必填”，还要让 schema 本身足够表达“还能传什么”
   - 如果工具的能力强依赖可选字段组合，`properties` 必须写全，描述也应提示典型用法
   - 否则 agent 只看见 `required`，会自己脑补参数名或漏掉关键控制字段

4. 单工具文件和工具总表必须保持一致
   - 如果同时维护 `data/tool_schemas/device/*.yaml` 和 `data/tool_schemas/mobile_assistant_mcp_tools.yaml`
   - 那么字段名、`required`、description、返回结构都要同步
   - 不允许一个版本接受别名、另一个版本不接受，或一个版本语义更严格、另一个更宽松

5. 对“高风险动作”或“需要白名单/确认/时长”的写工具，schema 应直接暴露这些控制杆
   - 例如白名单联系人、是否允许消息/来电、是否静音其他通知、持续时间、确认开关
   - 不要把这些关键控制条件只埋在 scenario 文案里，而 schema 本身看不出来

---

## 三、当前评测框架的关键约束

### 1. `allowed_tools` 必须写完整 key

格式必须是：

- `device:weather.get_forecast`
- `device:maps.get_routes`

不是只写 `weather.get_forecast`。

---

### 2. `mock world` 的重要限制

当前 `MockWorld` 的行为：

- `maps.get_routes`
- `maps.start_navigation`
- `maps.get_navigation_status`
- `calendar.get_next_event`

有专门 handler。

除此之外：

- 只读工具默认从 `world.state.tool_responses[tool_name]` 取静态返回
- 写工具默认把调用记到 `operation_log`
- 如果 `world.state.tool_state_updates[tool_name]` 存在，会自动 merge 到 state

因此你要记住：

### 2.1 对只读工具

如果它走 generic path，你必须在 `world.state.tool_responses` 里给出返回值。

例如：

```yaml
world:
  state:
    tool_responses:
      learning.get_course_overview:
        course:
          title: 数据分析基础
          progress_completed: 18
          progress_total: 32
```

### 2.2 对写工具

如果希望写工具改状态，你必须提供：

```yaml
world:
  state:
    tool_responses:
      reservations.cancel:
        cancelled: true
        reservation_id: res_123
    tool_state_updates:
      reservations.cancel:
        reservations:
          current:
            status: cancelled
```

### 2.3 避免 mock world 无法区分多参数调用

generic 只读工具不会按参数分叉返回。

所以如果同一个 scenario 里必须：

- 对同一个工具查两个城市
- 对同一个工具查两个对象
- 同一工具多次调用且每次结果必须不同

优先用以下方案之一：

1. 设计成一次工具调用就返回两个对象
   - 例如 `weather.compare_cities`
2. 使用已有专门 handler 的工具
3. 只有在确实必要时，新增 world handler

默认优先方案 1，不要轻易要求改 runtime 代码。

---

## 四、scenario 设计原则

### 1. `user_prompt` 要自然、短、略欠完整

应像真实用户说的话，不要把隐藏规则直接写出来。

语言也要控制：

1. 默认把 `user_prompt` 写成自然中文口语
   - 即使 `final_version` 或 `original_query` 是英文，也优先翻成简洁自然的中文表达
   - 目标是让整套 scenario 的作者文本风格一致，避免批量产出里大面积英文 `user_prompt`

2. 只有在以下情况，才保留英文 `user_prompt`
   - 英文措辞本身就是题目语义的一部分
   - 品牌、片名、游戏名、产品名、固定 UI 文案必须保留英文
   - 你明确希望保留“英文用户真实说法”作为考点

3. 即使保留英文，也只保留 `user_prompt` 这一小段
   - `implicit_reason`、`hidden_constraints`、`suggested_plan`、`execution_rules`、`notes` 仍默认写中文
   - 不要出现“作者说明是中文，但 `user_prompt` 大批量都是英文模板句”的退化

4. 一个实用检查：
   - 如果连续生成 10 条里有超过 2 条英文 `user_prompt`
   - 说明你已经过度依赖原 query 文本，应该主动改回中文

好例子：

- `快到家的时候帮我把家里的空调和热水器提前打开。`
- `帮我找一集时长刚好覆盖通勤时间的播客，并直接开始播放。`

坏例子：

- `请先查询 ETA，并在 ETA 小于 15 分钟时创建自动化。`

---

### 1.1 这些 scenario 是“考题”，不是“工具对照表”

请始终记住一个核心目标：

这些 scenario 的价值，不是在于给模型一个“名字刚好匹配需求”的工具，
而是在于检验模型能否：

- 理解用户真实目标，而不是只看字面动作
- 在不完全贴合的工具集合中，主动选择合适的 MCP 工具链
- 通过读取证据、补足隐式约束、再决定是否执行动作，完成任务

因此你在设计 scenario 时，要刻意避免把题目做成“看到工具名就等于看到答案”的形式。

具体要求：

1. 不要为了让题目更顺手，就发明一个和用户目标几乎同名、一步到位的工具
   - 例如用户想“判断是否该取消婚礼”，不要新增 `event.should_cancel_wedding`
   - 应优先提供天气、预订状态、取消规则、备选方案之类的中间证据工具

2. `allowed_tools` 可以和真实世界的理想工具集合不完全一致
   - 只要这些工具仍然足以让强模型通过正确理解和组合来完成任务
   - 重点是“能推理并完成”，不是“接口刚好贴脸”

3. 优先保留“间接完成任务”的结构
   - 用户要的是取消 / 判断 / 推荐 / 安排
   - 但工具更适合提供状态、规则、上下文、候选项、执行入口
   - 让 agent 自己把这些中间信息转成最终动作或结论

4. 如果某个 scenario 只要看一眼工具名就能直接定位唯一答案路径，通常说明题目做得太浅
   - 更好的题目应要求 agent 至少跨一步：
   - 例如先查状态，再查规则，再决定是否执行
   - 或先查候选，再结合偏好/风险做选择

5. 但“间接”不等于“故意刁难”
   - 不要把关键能力完全藏掉，导致题目事实上不可解
   - 正确目标是：工具不完美贴合，但推理后可解

6. 一个实用判断标准：
   - 如果你设计完后，感觉这个 scenario 更像“工具调用演示”
   - 而不是“隐式问题求解”
   - 那就应该回去削弱工具与答案之间的直接对应关系

---

### 2. `implicit_reason` 写“表面请求 vs 真实目标”

模板：

```text
用户表面上是在请求 {surface_action}，但真实目标是 {real_goal}。
系统需要先确认 {condition_a}，再结合 {condition_b} 做判断或执行，
且不能忽略 {risk_or_buffer}。
```

---

### 3. `hidden_constraints` 只写真正决定成败的约束

通常 3 到 5 条即可。

好约束：

- 需要先确认是否真的尚未发货。
- 应基于 ETA 阈值触发，而不是固定时间触发。
- 删除前需要强制二次确认。

坏约束：

- 需要认真思考。
- 要给出高质量回答。

---

### 4. `execution_rules` 必须可执行、可验证

它是 world/evaluator 的 contract，不是作者随笔。

好例子：

- 应先检查订单状态，再决定是否取消。
- 若预订不可退款，则默认不能直接取消。
- 最终输出应给出结论与关键原因，而不是只返回原始工具结果。

---

### 5. 优先复用已有工具，确实缺口再新增

优先检查：

- `data/tool_schemas/device/*.yaml`
- `data/tool_schemas/mobile_assistant_mcp_tools.yaml`

新增工具的标准：

- 该能力在多个场景中都可能复用
- 不是把“最终答案”直接封进工具
- 返回结构化中间证据，不直接替 agent 完成推理

坏工具：

- `decide_best_city_for_outdoor_wedding`
- `should_cancel_the_event`

好工具：

- `weather.compare_cities`
- `reservations.get_details`
- `training.get_training_overview`

### 5.1 选工具时，优先补齐“观察层 / 判断层 / 执行层”

在实际写 `allowed_tools` 前，先问自己这条 query 属于哪一种：

1. 纯观察型
   - 目标只是查事实并返回
   - 这时可以只放观察工具

2. 观察 + 判断型
   - 目标是基于多个证据做结论
   - 这时 `allowed_tools` 至少要覆盖所有关键证据来源

3. 观察 + 判断 + 执行型
   - 目标是先查、再判断、最后落动作
   - 这时不能只放观察工具，也不能只放执行工具

尤其注意下面这些常见组合：

- 取消类：
  - 常见需要 `状态/规则读取工具 + 取消执行工具`
- 时间协调类：
  - 常见需要 `时间/时区/日程读取工具 + 提醒/安排执行工具`
- 天气决策类：
  - 常见需要 `天气工具 + 与真实后果相关的工具`
  - 例如路线、预订、活动、提醒、备选方案
- 推荐 / 选择类：
  - 常见需要 `候选集合工具 + 用户偏好 / 上下文工具`

如果你发现自己写出来的 `allowed_tools` 只有 1 个，而 query 明显属于上面第 2 或第 3 类，请重新设计。

---

### 6. world.state 应尽量支持 hidden constraints

不要只写表层数据。

例如用户问“取消预订”，state 不应只有：

```yaml
reservation:
  status: active
```

更好的是：

```yaml
reservations:
  current:
    reservation_id: res_123
    status: active
    refundable: false
    cancellation_deadline: "18:00"
    penalty_amount: 120
    alternatives:
      - reschedule
      - voucher
```

---

### 7. 评测优先可跑，再追求完美 rubric

当前项目允许：

- `evaluation_rubric: []`

此时 LLM evaluator 会根据：

- `implicit_reason`
- `hidden_constraints`
- `suggested_plan`
- `execution_rules`

自动推导 criteria。

如果某个场景非常容易写 deterministic rubric，就补 2 到 4 条；
否则先保持空数组，不要为了凑 rubric 写低质量检查。

---

## 五、从 query 到 scenario 的标准流程

### Step 1. 识别真实目标

不要停留在 query 字面动作。

例如：

- `查询天气` 的真实目标可能是 `判断是否取消活动`
- `找提示词` 的真实目标可能是 `完成用户故事拆解`
- `取消预订` 的真实目标可能是 `退出当前安排并控制损失`

### Step 2. 识别必须外显到 world 的证据

问自己：

- agent 需要查哪些事实？
- 这些事实当前能否被工具读取？
- 哪些事实属于高风险/截止时间/冲突源？

### Step 3. 选工具

- 能复用现有工具就复用
- 缺能力就新增小而通用的工具
- 不要把推理封进工具

### Step 4. 设计 world.state

- 给足 agent 做判断所需的证据
- 给足 evaluator 回溯成败原因的依据
- 对写工具补 `tool_state_updates`

### Step 5. 写 `execution_rules`

把隐藏约束改写成可执行 contract。

### Step 6. 决定是否写 rubric

- 容易 deterministic 检查就写
- 不容易就先空

---

## 六、输出要求

当你实际执行该 prompt 时，请输出一个 JSON 对象，字段如下：

```json
{
  "scenario_file_name": "",
  "scenario_yaml": "",
  "sample_result": {
    "tool_trace": [
      {
        "step": 1,
        "tool": "device:some.tool",
        "arguments": {},
        "expected_observation": ""
      }
    ],
    "final_answer": ""
  },
  "new_tool_schema_files": [
    {
      "file_name": "",
      "yaml": ""
    }
  ],
  "notes": ""
}
```

要求：

- `scenario_yaml` 是完整 YAML 文本
- `sample_result` 必须与 `scenario_yaml` 中的作者字段 `sample_result` 一致，不允许两份内容互相矛盾
- `new_tool_schema_files` 可以为空数组
- 不要输出 markdown fence
- 文件名使用 snake_case
- 除 `allowed_tools`、tool schema 中的 `server/tool_name`、以及 `world.context / world.state` 中必须保留的英文标识、ID、地名、时间字符串、路径等客观字段外，其他作者性文本尽量写成中文
- 尤其是 `implicit_reason`、`hidden_constraints`、`suggested_plan`、`execution_rules`、`notes`、tool `description`，默认优先中文
- `user_prompt` 默认也优先写成自然中文，不要因为原 query 是英文就机械保留英文
- 只有当英文措辞本身是题目语义组成部分时，才保留英文 `user_prompt`
- 如果保留英文 `user_prompt`，请在 `notes` 里简短说明为什么必须保留英文
- 对 `allowed_tools` 做交付前自检：如果题目属于取消 / 推荐 / 计划 / 冲突消解类，且工具数少于 3，默认视为设计不足，需重写
- 对“多候选中定位目标”的题，若没有候选列表类工具，默认视为设计不足，需重写
- 对“先判断再决定是否执行”的题，若没有执行工具，默认视为设计不足，需重写
- 对 `sample_result.tool_trace` 做交付前自检：其中每个工具都必须出现在 `allowed_tools` 里
- 对 `sample_result.final_answer` 做交付前自检：必须能被当前 `world.state`、`execution_rules`、`evaluation_rubric` 支撑
- 交付前必须完整检查题目是否合理：如果发现工具链断裂、证据不足、正确答案与规则不一致、或 evaluator 无法稳定判断，就先调整 scenario，而不是原样输出

---

## 七、批量处理时的附加规则

对于 `top100` 批量 query：

1. 每次只处理 10 条。
2. 同一批次尽量复用新工具，不要重复发明同义工具。
3. 如果一个 query 和已有 scenario 高度同构，允许复用同一工具组合，但 `world.state` 与 `user_prompt` 必须重写。
4. 每批完成后，检查：
   - scenario id 是否唯一
   - allowed_tools 是否都存在于 registry
   - sample_result 中引用的工具是否全部存在于 allowed_tools
   - sample_result 的关键证据是否都能在 world / tool_responses / tool_state_updates 中找到支撑
   - 所有新增写工具是否补了 `tool_state_updates`
   - 是否有把隐藏约束泄露到 `user_prompt`
   - 若题目不合理，是否已经在交付前重写或调整，而不是把问题留到评测阶段

---

## 八、与当前 top100 的关系

对于来自 `implicit_query_top100_drop100_cancel.jsonl` 的条目：

- 优先使用 `final_version` 作为 scenario 种子
- 用 `original_query` 补关键上下文
- 用 `primary_rule_family / secondary_rule_families` 指导主规则选择
- 用 `implicit_points` 辅助写 `hidden_constraints`
- 可回查 `source_file + source_id` 中的 `available_tools / full_context`
  来理解原始任务依赖的工具类型，但不要直接复制原工具名

你的最终目标是：

把一个高质量 implicit query 转成一个能在 ImplicitConstraints 当前评测框架里执行的、结构一致的 scenario。


已删除：hidden_constraints、execution_rules、llm_evaluation_instructions、evaluation_rubric: []。
保留：implicit_reason、suggested_plan、sample_result、world 等（给跑场景用）；评测不再读这些字段。



时间/截止/缓冲/时区/ETA
先查真实状态再行动
风险防呆/确认后执行
排序/筛选/最优选择
跨对象依赖/整体损失或整体可行性
个性化/关系/偏好识别
自动化/提醒/真正落地执行

## 九、implicit_eval_points 标签与最终打分

从现在开始，每一条 `implicit_eval_points` 都必须以一个规则家族标签开头，格式固定为：

- `【时间/截止/缓冲/时区/ETA】...`
- `【先查真实状态再行动】...`
- `【风险防呆/确认后执行】...`
- `【排序/筛选/最优选择】...`
- `【跨对象依赖/整体损失或整体可行性】...`
- `【个性化/关系/偏好识别】...`
- `【自动化/提醒/真正落地执行】...`

要求：

1. 不再使用自定义小标题式标签，如 `【缓冲】`、`【可恢复性】`、`【先查天气】`。
2. 每个 point 只能归到上述 7 类中的 1 类。
3. 同一 scenario 可以出现多个来自同一规则家族的 points。
4. point 文案主体仍写具体评测要求，但 `【】` 内只能放规则家族名。
5. 若某个旧 point 同时涉及多个维度，优先标它最主要、最决定通过与否的那一类。

最终批量打分只保留 3 组结果：

1. `全对题目数 / 题目总数` 的百分比分数。
2. `通过 points 数 / points 总数` 的百分比分数。
3. 按上述 7 个规则家族分别统计：`该规则下通过的 points 数 / 该规则下 points 总数`。

不要再输出额外的平均分、重复二值分或其他派生汇总口径。



该问清还是该直接做
隐私/权限/最小暴露
工具结果冲突/信息源冲突
失败/空结果/降级策略
预算/价格/风险偏好
长期条件监控/后续回滚
社交语境/沟通语气
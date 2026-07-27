# Kimi K2.6 Protocol III 真实网站评测

评测日期：2026-07-26（Asia/Shanghai）

## 结论

本轮对固定种子 `20260726` 的 8 条分层样本进行了零任务重试的真实网站评测。8 条任务均启动 Agent 并访问目标网站，但没有任务产生通过 `CompletionVerifier` 的 `agent_answer`。

- exact accuracy（全量分母）：`0 / 8 = 0%`
- normalized accuracy（全量分母）：`0 / 8 = 0%`
- 验证完成率：`0 / 8 = 0%`
- 可比较答案数：`0`
- exact/normalized accuracy（仅可比较答案）：`N/A`，不能计算 `0 / 0`

这是本地 Chrome/CDP、当前 Agent、Moonshot API 配额共同作用下的端到端结果，不应解释为 Kimi K2.6 在已完成答案上的语义正确率。7 条任务因 Moonshot 组织级 `3,000,000 TPM` 限额中断，另外 1 条在 100 步内未完成。

## 配置与口径

- 数据源：`/Users/admin/Desktop/webRetriever/data/protocol3.json`，共 100 条
- 样本：`evaluation/protocol3_sample_8.json`
- 抽样：确定性分层抽样，seed `20260726`
- 模型：Moonshot `/v1/models` 实际返回的 `kimi-k2.6`
- API：`https://api.moonshot.cn/v1/chat/completions`
- 浏览器：本机 Google Chrome headless，通过独立 CDP URL 连接
- 浏览器执行：只使用 Playwright `connect_over_cdp` 和 Playwright 页面操作
- 并发：SEC 单任务预检后，其余 7 条使用 7 个隔离 Worker/CDP 并发
- 最大步数：每任务 100
- 重试：任务级 0 次；连接前即失败的 Chrome 生命周期预检不计入执行任务
- 评分：`result.json.agent_answer` 与数据集 `answer` 做 exact 和 NFKC/大小写/空白归一化对比；失败和空答案按全量分母计 0

Kimi K2.6 开启 thinking 时不接受 `tool_choice=required`，关闭 thinking 后只接受 `temperature=0.6`。本轮在真实任务前已用结构化函数工具探测确认该组合可用；代码仅对 `kimi-k2.6` 使用这组参数，其他模型继续使用 `temperature=0`。

## 逐题结果

| task_idx | 网站 | 状态 | 工具步数 | 结果 |
|---:|---|---|---:|---|
| 5 | data.gov.hk | `FAIL_AGENT_ERROR` | 7 | Moonshot 429 TPM 限额 |
| 17 | shanghairanking.cn | `FAIL_AGENT_ERROR` | 14 | Moonshot 429 TPM 限额 |
| 25 | ncss.cn | `FAIL_AGENT_ERROR` | 6 | Moonshot 429 TPM 限额 |
| 37 | sec.gov | `FAIL_MAX_STEPS` | 100 | 49 次 observe、45 次 tabs、6 次 wait，未通过 finish |
| 40 | earthquake.usgs.gov | `FAIL_AGENT_ERROR` | 12 | Moonshot 429 TPM 限额 |
| 63 | wid.world | `FAIL_AGENT_ERROR` | 7 | Moonshot 429 TPM 限额 |
| 68 | pageviews.wmcloud.org | `FAIL_AGENT_ERROR` | 8 | Moonshot 429 TPM 限额 |
| 75 | gs.statcounter.com | `FAIL_AGENT_ERROR` | 8 | Moonshot 429 TPM 限额 |

SEC 任务实际访问了 SEC 首页、EDGAR Search、NVIDIA CIK 页面和带 `type=10-K` 的公司申报列表 URL。模型 API 请求均返回 200，但 Agent 在多个标签页间反复观察，最终达到 100 步；未生成未经验证的 SUCCESS。

其余任务在触发 429 前均已访问目标网站并执行了 Playwright 动作。例如 task 5 进入香港数据集详情页，task 17 进入 2024 哲学专业排名页，task 40 进入 USGS Earthquake Catalog，task 68 已开始填写 Pageviews 表单。

## 发现与修复

1. 当前代码固定 `temperature=0`，Kimi K2.6 会返回 400；已增加 Kimi 专用的 `temperature=0.6` 和 `thinking=disabled` 配置及单测。
2. 真实站点出现 gzip/二进制 XHR 请求体，Playwright 的 `request.post_data` 会尝试 UTF-8 解码并在事件监听器中抛出 `UnicodeDecodeError`。本轮结束后已改用 `post_data_buffer`；结构化 JSON/URL-encoded 数据继续脱敏，二进制或非结构化正文只保留长度、SHA-256 和截断标记，并增加回归测试。
3. 7 Worker 并发会让该 Moonshot 账号超过 3,000,000 TPM。Protocol III 允许最多 8 Worker，但模型服务配额不足以支撑当前大上下文下的 7 并发。后续评测应使用 1 Worker 顺序运行或先提升 TPM 配额；本轮按零任务重试原则未重新执行失败任务。

## 产物

- 脱敏对比报告：`evaluation/kimi_k2_6_sample8_comparison.json`
- 原始本地轨迹：`test_results/kimi-k2.6-sample8-20260726/`（被 `.gitignore` 排除）
- 每任务产物：`result.json`、`evidence.json`、`capture.json`、`trajectory/`、`trajectory_visual/`、`observations/`

项目内轨迹约 44 MB。`test_results/` 中的 `result.json`、summary 和日志已清除 Moonshot 组织、项目和访问密钥前缀等提供商标识，同时保留状态码、限额类型、工具轨迹和网页证据。`logs/summary.json` 是后执行的 Stage 2 七任务摘要，因为两阶段共用输出目录时该文件会被覆盖；完整 8 条计数以八个任务目录和 `kimi_k2_6_sample8_comparison.json` 为准。

首次 CDP 生命周期预检的审计副本位于 `test_results/kimi-k2.6-preflight-invalid-cdp-20260726/`。该次在 Worker 调用 `connect_over_cdp` 时收到 `ECONNREFUSED`，`predict_length=0`、动作数为 0、URL 数为 0，日志中没有 Moonshot 模型请求，因此排除在任务执行和准确率分母之外。预检随后通过将 Chrome 与 Agent 放入同一进程会话解决。

## 自动化验证

- Moonshot `/v1/models`：实际确认账号可用模型包含 `kimi-k2.6`
- 结构化工具探测：`tool_choice=required`、`parallel_tool_calls=false`、`temperature=0.6`、`thinking.disabled` 返回合法函数调用
- 图片输入探测：1x1 本地 PNG data URI 返回 200，响应包含内容；没有访问第三方图片
- 完整测试：`WEB_AGENT_REQUIRE_CHROME=1 .venv/bin/pytest -q`
- 完整测试结果：`49 passed in 41.84s`
- focused 测试：Kimi 参数、敏感/二进制请求体、事件监听容错和 BrowserActor 集成测试通过

响应正文方面，只读取声明了可信 `Content-Length`、未压缩且不超过采集上限的 XHR/Fetch；未知长度、压缩或超限正文直接跳过。读取后的正文只保留可完整解析并按字段脱敏的 JSON/URL-encoded 数据，其他内容只记录长度和 SHA-256。这避免 Playwright 对未知大小正文做全量读取，也修正了截断 JSON 泄密和“长度恰好等于上限”误报截断的问题；代价是部分 text、chunked 或压缩响应只能作为 URL/状态/头部元数据证据。

## 实际命令

密钥仅通过 shell 环境变量注入，未写入仓库：

```bash
jq '[.[] | select(.task_idx == 37)]' \
  /Users/admin/Desktop/webRetriever/data/protocol3.json \
  > /tmp/protocol3_kimi_stage_01.json

jq '[.[] | select(.task_idx == 5 or .task_idx == 17 or .task_idx == 25 or
  .task_idx == 40 or .task_idx == 63 or .task_idx == 68 or .task_idx == 75)]' \
  /Users/admin/Desktop/webRetriever/data/protocol3.json \
  > /tmp/protocol3_kimi_stage_02.json

.venv/bin/python -m web_agent.cli \
  --input /tmp/protocol3_kimi_stage_01.json \
  --output test_results/kimi-k2.6-sample8-20260726 \
  --cdp_url "$CDP_0" \
  --model kimi-k2.6 \
  --api_base https://api.moonshot.cn/v1 \
  --api_key "$MOONSHOT_API_KEY" \
  --max_steps 100

.venv/bin/python -m web_agent.cli \
  --input /tmp/protocol3_kimi_stage_02.json \
  --output test_results/kimi-k2.6-sample8-20260726 \
  --cdp_url "$CDP_0" "$CDP_1" "$CDP_2" "$CDP_3" "$CDP_4" "$CDP_5" "$CDP_6" \
  --model kimi-k2.6 \
  --api_base https://api.moonshot.cn/v1 \
  --api_key "$MOONSHOT_API_KEY" \
  --max_steps 100

.venv/bin/python -m web_agent.protocol3_eval compare \
  --input evaluation/protocol3_sample_8.json \
  --results test_results/kimi-k2.6-sample8-20260726 \
  --output evaluation/kimi_k2_6_sample8_comparison.json
```

本轮没有运行 Protocol III 全量 100 条，因此不能声称 100 条数据集或赛事总体准确率为 0%。8 条样本偏能力压力测试，也不足以给出窄置信区间。

# 验收记录

记录日期：2026-08-03（Asia/Shanghai）。本文件只记录本工作区实际执行过的命令与结果。

## 环境

- Python 3.13.12
- Playwright Python 1.61.0
- Google Chrome 150.0.7871.184
- OpenAI Agents Python 0.18.3
- OpenAI Python 2.48.0
- Poppler `pdftoppm` 26.03.0

赛事公开规则尚未固定 Playwright 版本，因此当前由 `pyproject.toml` 独立约束为
`playwright>=1.49,<2`；正式模板发布后再切换到模板版本。

## 自动化测试

完整单元测试与真实 Chrome/CDP 集成测试：

```bash
WEB_AGENT_REQUIRE_CHROME=1 .venv/bin/pytest -q
```

实际结果：`207 passed in 45.07s`。其中 11 个集成用例通过 Playwright
`connect_over_cdp()` 连接临时 Chrome，未使用裸 CDP 客户端；其余 196 个用例覆盖契约、
验证器、运行指纹与安全续跑、真实 preflight、动作 deadline/迟到隔离、Worker watchdog、
证据脱敏、进程/线程隔离和评测入口。

单独运行非浏览器用例：

```bash
.venv/bin/pytest -q -m 'not integration'
```

实际结果：`196 passed, 11 deselected in 4.37s`。

## 构建与静态自检

```bash
.venv/bin/python -m compileall -q src tests
node --check vendor/browsergym/src/browsergym/core/javascript/frame_mark_elements.js
node --check vendor/browsergym/src/browsergym/core/javascript/frame_unmark_elements.js
uv build --wheel --out-dir /tmp/webretriever-wheel-final-20260803 .
.venv/bin/python -m web_agent.cli --help
.venv/bin/python -m web_agent.cli \
  --preflight --output /tmp/webretriever-preflight-smoke-final
```

Python、两份 JavaScript 语法、CLI help 和 wheel 构建均通过。无 CDP URL 的失败烟测返回退出码 2
及稳定错误码 `PREFLIGHT_CDP_WORKER_COUNT`，同时落盘审计结果并保持 `model_requests: 0`、
`target_navigation: false`；真实 CDP 能力由上面的 Chrome 集成测试覆盖。

另在 `/tmp` 新建干净 Python 3.12 venv 安装该 wheel；安装态 `web_agent` 与 vendored
`browsergym` 均可导入，运行源码哈希非空；修改临时安装态 BrowserGym JavaScript 后哈希随之变化，
确认 wheel 安装场景不会退化为空源码哈希。

当前主机没有 `docker` 命令，因此未声称 Docker 镜像或 Compose 已实际构建。Dockerfile 已提供
非 root 用户、外部 CDP 约束、健康检查和扫描 PDF 所需的 `poppler-utils`，应在具备 Docker 的
交付环境按 `DEPLOYMENT.md` 复验。

## Protocol III 数据与报告

实际输入：`/Users/admin/Desktop/webRetriever/data/protocol3.json`。

```bash
.venv/bin/python -m web_agent.protocol3_eval profile \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --output evaluation/protocol3_profile.json

.venv/bin/python -m web_agent.protocol3_eval sample \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --output evaluation/protocol3_sample_8.json \
  --report evaluation/protocol3_sample_8_report.json \
  --size 8 --seed 20260726

.venv/bin/python -m web_agent.protocol3_eval compare \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --results evaluation/not_run \
  --output evaluation/protocol3_comparison_not_run.json \
  --csv evaluation/protocol3_comparison_not_run.csv \
  --default-not-run-reason '当前环境未提供真实网站运行所需的模型 API 凭证；未执行任何 Protocol III 真实网站任务'
```

实际剖析：100 条任务、98 个 website 值、97 个 host；5 个要求字段均 100% 存在；答案均为
字符串，其中标量 77 条、多行 23 条。任务分层为 analytical 72、exhaustive 15、lookup 13。
固定种子样本索引为 `5, 17, 25, 37, 40, 63, 68, 75`。

当前环境未提供可调用的模型 API 凭证，因此真实网站任务实际运行数为 0；对比报告明确记录
`executed_tasks=0`、`not_run_tasks=100`、`comparable_tasks=0`，未报告任何真实站点准确率。

# 手工验收

本文只描述可复现的本地夹具与显式运行命令。真实网站是否可访问、是否需要凭证，以及网站任务是否实际执行，必须以当次输出目录为准；不得把数据剖析、离线答案对比或本地夹具通过表述成真实网站成绩。

## 1. 安装

要求 Python 3.10 以上、Chrome 或 Chromium，以及可用的 OpenAI 兼容模型 API。项目不要求安装 `browsergym-core`，Playwright 版本由本项目独立管理。

扫描 PDF 的局部视觉降级还要求 Poppler：macOS 使用 `brew install poppler`，Debian/Ubuntu
使用 `sudo apt-get install poppler-utils`。Docker 镜像已安装 `poppler-utils`。

```bash
cd /Users/admin/Desktop/webRetriever/WebDeepRetriever
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

本地和正式 Agent 都通过 Playwright `connect_over_cdp()` 连接已有浏览器。不要执行 `playwright install` 来替代外部 CDP 浏览器，也不要使用 UI-TARS、AnySearch、外部搜索引擎、裸 CDP 客户端或主动网站 HTTP/API 请求。

## 2. 启动本地 Chrome CDP

为每个 worker 使用独立端口和独立用户目录。下面只启动浏览器，不访问目标网站。

macOS：

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/webretriever-chrome-9222 \
  --no-first-run about:blank
```

Linux：

```bash
google-chrome \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/webretriever-chrome-9222 \
  --no-first-run about:blank
```

用 Playwright 验证连接，避免用其他 CDP 客户端：

```bash
.venv/bin/python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
    print({"connected": browser.is_connected(), "contexts": len(browser.contexts)})
PY
```

## 3. 自动化夹具验收

完整测试：

```bash
WEB_AGENT_REQUIRE_CHROME=1 .venv/bin/python -m pytest -q
```

只跑无需 Chrome 的单元测试：

```bash
.venv/bin/python -m pytest -q -m 'not integration'
```

Playwright/CDP 集成夹具会自行启动本机 Chrome、启动仅绑定 `127.0.0.1` 的确定性网页服务并在结束后回收：

```bash
WEB_AGENT_REQUIRE_CHROME=1 .venv/bin/python -m pytest -q tests/integration/test_browser_actor.py
```

逐项验收命令与通过条件：

| 能力 | 命令选择器 | 通过条件 |
| --- | --- | --- |
| 原生/自定义表单、重复标签、陈旧 bid、SPA 延迟 | `::test_native_custom_form_duplicate_labels_stale_bid_and_spa` | `fill/select/set_checked` 回读正确；自定义 option 重观察；旧 bid 明确失败 |
| 嵌套 iframe、开放 Shadow DOM | `::test_nested_iframe_and_open_shadow_dom` | 深层控件和 Shadow 控件均有 bid 且可由 Locator 操作 |
| 分页、虚拟列表 | `::test_pagination_and_virtual_list_exhaustion` | 下一页禁用并覆盖 6 项；虚拟列表终止后保持 15 项无新增 |
| Canvas、图片、弹窗、新标签页 | `::test_canvas_image_dialog_and_new_tab` | 只裁剪目标视觉区域；dialog 有回执；新标签可跟随和切回 |
| 上传、下载、PDF、浏览器触发网络响应 | `::test_upload_download_pdf_document_and_browser_network` | 文件回读、PDF 文本、XHR/Fetch 证据及敏感字段脱敏均正确 |
| BrowserActor 线程亲和 | `::test_browser_actor_serializes_playwright_on_owner_thread` | 所有 Playwright 调用固定在唯一 owner thread |

示例：

```bash
.venv/bin/python -m pytest -q \
  tests/integration/test_browser_actor.py::test_nested_iframe_and_open_shadow_dom
```

DOM 可定位控件的测试中不应出现坐标点击接口。局部视觉检查只能用于 DOM、ARIA、页面结构化数据、浏览器触发的响应和文档文本均无法表达的 Canvas、图片、图表或扫描 PDF；截图仍应逐步落盘，但默认不发送给决策模型。

## 4. 运行 Agent

先从完整数据集中生成小样本；这一步不访问任何网站：

```bash
.venv/bin/python -m web_agent.protocol3_eval sample \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --output /tmp/protocol3.sample.json \
  --report /tmp/protocol3.sample.report.json \
  --size 8 --seed 20260726
```

确认 Chrome、模型 API 和样本范围后运行。每个 CDP URL 对应一个独立 worker，允许 1 到 8 个 URL，单任务最多 100 步：

```bash
OPENAI_API_KEY='替换为实际密钥' \
.venv/bin/python -m web_agent.cli \
  --input /tmp/protocol3.sample.json \
  --output /tmp/webretriever-output \
  --cdp_url http://127.0.0.1:9222 \
  --model gpt-4.1-mini \
  --api_base https://api.openai.com/v1 \
  --max_steps 100
```

在没有网站访问授权、凭证或可用模型 API 时不要执行该命令，并在离线报告的未运行原因文件中如实记录原因。

## 5. Protocol III 离线评测

字段剖析：

```bash
.venv/bin/python -m web_agent.protocol3_eval profile \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --output /tmp/protocol3.profile.json
```

答案对比仅读取标准答案和已经落盘的 `result.json.agent_answer`，不会访问网站：

```bash
.venv/bin/python -m web_agent.protocol3_eval compare \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --results /tmp/webretriever-output \
  --unrun-reasons /tmp/unrun-reasons.json \
  --output /tmp/protocol3.comparison.json \
  --csv /tmp/protocol3.comparison.csv
```

未运行原因可按 `task_id` 写成对象：

```json
{
  "330dcf5ea8084f328316f90f0eeac040": "目标站点要求当前环境未提供的账号凭证"
}
```

报告分别统计 `executed_tasks`、`successful_results`、`comparable_tasks`、exact 和 normalized；`NOT_RUN` 进入全量分母但不伪装成执行结果。normalized 只做 Unicode NFKC、大小写和空白规范化，不改变列表顺序或数值类型。

## 6. 结果文件检查

每个任务目录至少检查：

- `result.json`：核心字段为 `agent_answer`。只有 `status == "SUCCESS"` 且 CompletionVerifier 已接受 finish，才能认定成功。
- `trajectory/` 与 `trajectory_visual/`：逐步审计截图。
- `capture.json`：只包含页面通过浏览器触发的有界 XHR/Fetch，敏感头和字段应脱敏。
- `actions`、`thoughts`、`urls`、回执与证据：位于结果或相应审计文件中，且答案字段需绑定访问过 URL 的证据。
- “全部/完整/列出”等任务：必须有可验证的 `coverage`，并证明下一页、游标、声明总数或虚拟列表已经终止。

可用下面的只读检查统计输出，命令不会访问网站：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/tmp/webretriever-output')
results = [json.loads(p.read_text(encoding='utf-8')) for p in root.rglob('result.json')]
print({
    'result_files': len(results),
    'verified_success': sum(r.get('status') == 'SUCCESS' and r.get('agent_answer') is not None for r in results),
    'missing_answer': sum(r.get('agent_answer') is None for r in results),
})
PY
```

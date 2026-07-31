# 部署交付

当前赛事方尚未发布固定容器模板，因此默认交付为 Docker。正式模板发布后应以模板锁定的基础镜像和 Playwright 版本为准，同时保留本项目的 BrowserGym vendoring、验证协议和输出格式。

## 运行边界

- 正式容器只接收比赛方提供的外部 CDP URL；镜像不下载、不安装、不启动浏览器。
- 所有浏览器连接与交互都经 Playwright，禁止裸 CDP 客户端、`cdp-use`、UI-TARS、AnySearch、外部搜索引擎及绕过网页的主动 HTTP/API 请求。
- 每个 CDP URL 固定分配一个 BrowserActor worker。URL 数量必须为 1 到 8 且互不相同。
- `max_steps` 外层配置限制在 1 到 100；实际采用基础 30 次模型请求加可证明进展信用，绝对不超过 60；模型单次请求上限为 180 秒。
- 上传仅允许任务输出父目录或 `WEBRETRIEVER_UPLOAD_ROOTS` 白名单；Compose 默认只额外允许只读输入目录。
- `output` 必须持久化；正式结果以各任务 `result.json.agent_answer` 和 CompletionVerifier 状态为准。

## 构建与镜像自检

```bash
cd /Users/admin/Desktop/webRetriever/WebDeepRetriever
docker build -t webdeepretriever:protocol3 .
docker run --rm webdeepretriever:protocol3 --healthcheck
```

预期输出：

```json
{"status": "ok", "playwright_transport": "cdp", "max_workers": 8}
```

这个健康检查验证 Python 包、运行入口和依赖可导入，不探测真实网站，也不把 CDP 可达性或模型可用性伪报为任务成功。

## Docker Compose

准备环境变量：

```bash
cp .env.example .env
```

填写 `.env` 中的模型 API、1 到 8 个逗号分隔且互不相同的外部 CDP URL、输入文件和可写输出目录。若浏览器运行在 Docker 宿主机，macOS/Windows 使用 `host.docker.internal`；Linux Compose 已配置 `host-gateway`。远程比赛 CDP 则直接填写赛事 URL。

仅在本地 Docker 联调时，宿主机 Chrome 需要监听容器可达地址，例如增加 `--remote-debugging-address=0.0.0.0 --remote-allow-origins=*`，并用防火墙把 9222 端口限制在本机/容器网段。非容器本地运行应继续绑定 `127.0.0.1`；正式赛事环境直接使用比赛方 CDP，不自行启动浏览器。

启动批任务：

```bash
docker compose up --build --abort-on-container-exit
```

查看退出状态与输出：

```bash
docker compose ps -a
find "${WEBRETRIEVER_OUTPUT_HOST:-./output}" -name result.json -print
```

容器以 UID/GID `10001:10001` 运行。部署前必须确保宿主机输出目录可由该 UID 写入；输入以只读方式挂载。

## 直接运行容器

以下示例连接宿主机已存在的 Chrome CDP。命令不会在容器内启动 Chrome：

```bash
mkdir -p /tmp/webretriever-output
docker run --rm --init \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  -e WEBRETRIEVER_MODEL="${WEBRETRIEVER_MODEL:-gpt-4.1-mini}" \
  -e WEBRETRIEVER_CDP_URLS=http://host.docker.internal:9222 \
  -e WEBRETRIEVER_MAX_STEPS=100 \
  -v /tmp/protocol3.sample.json:/work/input/tasks.json:ro \
  -v /tmp/webretriever-output:/work/output \
  webdeepretriever:protocol3 \
  --input /work/input/tasks.json --output /work/output
```

多 worker 时只修改 `WEBRETRIEVER_CDP_URLS`，例如：

```text
http://browser-0:9222,http://browser-1:9222,http://browser-2:9222
```

不要让多个 worker 共享同一个 URL，否则会破坏浏览器状态隔离并被运行器拒绝。

## 非 Docker 启动

安装、启动本地 Chrome CDP、使用 Playwright 验证连接和运行完整测试的命令见 `MANUAL_TESTING.md`。最小启动命令如下：

```bash
cd /Users/admin/Desktop/webRetriever/WebDeepRetriever
.venv/bin/python -m web_agent.cli \
  --input /tmp/protocol3.sample.json \
  --output /tmp/webretriever-output \
  --cdp_url http://127.0.0.1:9222 \
  --model gpt-4.1-mini \
  --api_base https://api.openai.com/v1 \
  --api_key "$OPENAI_API_KEY" \
  --max_steps 100
```

## 输入输出契约

输入是 Protocol III JSON 数组，每项使用 `task_idx`、`task_id`、`website`、`task`；离线评测数据另含标准 `answer`。运行器按 `<task_idx>_<task_id>/` 保存任务结果并支持跳过已有、含非空答案的 SUCCESS 结果。

输出卷需要保留：

- `result.json` 及其 `agent_answer`、状态、证据绑定和 coverage；
- `trajectory/`、`trajectory_visual/`、`capture.json`；
- actions、thoughts、urls、动作回执、证据和 worker 日志。

进程退出不等于任务成功。浏览器断开、页面崩溃、自适应预算或 60 次硬上限耗尽、非法工具输出和 CompletionVerifier 拒绝都必须保留失败状态，不能产生未经验证的 SUCCESS。

## Protocol III 离线报告

容器主要用于 Agent 批任务；数据剖析、分层抽样和答案对比建议在宿主机执行：

```bash
.venv/bin/python -m web_agent.protocol3_eval profile \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --output /tmp/protocol3.profile.json

.venv/bin/python -m web_agent.protocol3_eval sample \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --output /tmp/protocol3.sample.json \
  --report /tmp/protocol3.sample.report.json \
  --size 8 --seed 20260726

.venv/bin/python -m web_agent.protocol3_eval compare \
  --input /Users/admin/Desktop/webRetriever/data/protocol3.json \
  --results /tmp/webretriever-output \
  --unrun-reasons /tmp/unrun-reasons.json \
  --output /tmp/protocol3.comparison.json \
  --csv /tmp/protocol3.comparison.csv
```

这些命令均不访问真实网站。对比报告会显式区分 NOT_RUN、已运行失败、SUCCESS 和可比较答案；只有当真实 Agent 命令确实执行并产生结果时，才可报告对应网站结果。

## 依赖许可与运维注意

项目 vendoring 的 BrowserGym 固定为提交 `9e779f087de9a65668b6974d11f9ce9816026e96`，使用 Apache-2.0；来源、文件清单与校验记录位于 `vendor/browsergym/SOURCE.md`，许可证位于 `vendor/browsergym/LICENSE`。OpenAI Agents Python 0.18.3 使用 MIT。完整第三方说明见 `THIRD_PARTY_NOTICES.md`。

部署环境需要允许容器访问外部 CDP 和配置的模型 API，输出卷应有足够空间保存逐步截图、下载和审计记录。CDP 凭证与模型密钥只能通过密钥管理或环境变量注入，不得写进任务、轨迹、镜像或日志。开放 Shadow DOM 已覆盖；closed Shadow DOM 无法由页面标准 DOM 接口遍历。真实站点的登录、区域限制、反自动化和临时不可用仍需逐任务如实记录。

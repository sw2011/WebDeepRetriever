from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from web_agent.browser_actor import BrowserActor
from web_agent.evidence import EvidenceStore


def _minimal_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(payload)


BASE_STYLE = """
<style>
  body { font-family: sans-serif; margin: 24px; }
  label, button, input, select, a { display: block; margin: 8px 0; }
  button, input, select, a { min-height: 28px; }
  [hidden] { display: none !important; }
</style>
"""


PAGES: dict[str, str] = {
    "/forms": f"""<!doctype html><html><head><title>Forms</title>{BASE_STYLE}</head><body>
      <h1>Form controls</h1>
      <section aria-label="Billing contact">
        <label for="primary">Contact</label><input id="primary" name="primary">
      </section>
      <section aria-label="Shipping contact">
        <label for="secondary">Contact</label><input id="secondary" name="secondary">
      </section>
      <label for="country">Country</label>
      <select id="country" name="country">
        <option value="us">United States</option><option value="ca">Canada</option>
      </select>
      <label><input type="checkbox" name="terms"> Accept terms</label>
      <button id="city-trigger" aria-haspopup="listbox" aria-expanded="false"
        onclick="document.getElementById('city-list').hidden=false; this.setAttribute('aria-expanded','true')">Choose city</button>
      <div id="city-list" role="listbox" aria-label="City choices" hidden>
        <button role="option" onclick="document.getElementById('chosen').textContent='City: Paris'; document.getElementById('city-list').hidden=true">Paris</button>
        <button role="option" onclick="document.getElementById('chosen').textContent='City: Tokyo'; document.getElementById('city-list').hidden=true">Tokyo</button>
      </div>
      <p id="chosen">No city</p>
      <input name="volatile" value="old">
      <button onclick="document.querySelector('[name=volatile]').outerHTML='<input name=volatile value=new>'">
        Replace field
      </button>
      <button onclick="setTimeout(() => document.getElementById('spa-result').textContent='SPA ready', 450)">Load delayed state</button>
      <p id="spa-result">Waiting</p>
    </body></html>""",
    "/frames": f"""<!doctype html><html><head><title>Frames and shadow</title>{BASE_STYLE}</head><body>
      <h1>Frame and shadow roots</h1>
      <iframe title="outer-frame" src="/frame-one" style="width:500px;height:220px"></iframe>
      <open-panel></open-panel>
      <script>
        customElements.define('open-panel', class extends HTMLElement {{
          connectedCallback() {{
            const root = this.attachShadow({{mode:'open'}});
            root.innerHTML = `<label>Shadow value <input name="shadow-value"></label>
              <button onclick="this.nextElementSibling.textContent='Shadow saved'">Save shadow</button>
              <output>Not saved</output>`;
          }}
        }});
      </script>
    </body></html>""",
    "/frame-one": f"""<!doctype html><html><head><title>Outer frame</title>{BASE_STYLE}</head><body>
      <h2>Outer frame content</h2>
      <iframe title="inner-frame" src="/frame-two" style="width:400px;height:120px"></iframe>
    </body></html>""",
    "/frame-two": f"""<!doctype html><html><head><title>Inner frame</title>{BASE_STYLE}</head><body>
      <label for="deep">Deep input</label><input id="deep" name="deep-input">
      <p>Nested frame marker</p>
    </body></html>""",
    "/collection": f"""<!doctype html><html><head><title>Collections</title>{BASE_STYLE}
      <style>#virtual {{height:120px;width:320px;overflow-y:auto;border:1px solid #333}}
      #virtual-spacer {{height:600px;position:relative}}
      .virtual-row {{height:40px;position:absolute;left:0;right:0}}</style></head><body>
      <h1>Collections</h1>
      <p id="page-label">Page 1 of 3</p><ul id="page-items"></ul>
      <button id="next-page" onclick="nextPage()">Next page</button>
      <h2>Virtual products</h2>
      <div id="virtual" role="list" aria-label="Virtual products"><div id="virtual-spacer"></div></div>
      <p id="virtual-status">More</p>
      <script>
        let page = 1;
        function renderPage() {{
          pageItems.innerHTML = Array.from({{length:2}}, (_,i) => `<li>Page item ${{(page-1)*2+i+1}}</li>`).join('');
          pageLabel.textContent = `Page ${{page}} of 3`;
          nextPageButton.disabled = page === 3;
        }}
        function nextPage() {{ if (page < 3) {{ page++; renderPage(); }} }}
        const pageItems = document.getElementById('page-items');
        const pageLabel = document.getElementById('page-label');
        const nextPageButton = document.getElementById('next-page');
        renderPage();
        const virtual = document.getElementById('virtual');
        const virtualSpacer = document.getElementById('virtual-spacer');
        const virtualStatus = document.getElementById('virtual-status');
        const virtualRows = Array.from({{length:5}}, () => {{
          const row = document.createElement('div'); row.className='virtual-row';
          row.setAttribute('role','listitem'); virtualSpacer.appendChild(row); return row;
        }});
        function renderVirtual() {{
          const start = Math.min(10, Math.floor(virtual.scrollTop / 40));
          virtualRows.forEach((row, offset) => {{
            const index = start + offset;
            row.style.top = `${{index * 40}}px`; row.textContent = `Virtual item ${{index + 1}}`;
          }});
          virtualStatus.textContent = start === 10 ? 'End of virtual list' : 'More';
        }}
        renderVirtual();
        virtual.addEventListener('scroll', renderVirtual);
      </script><div style="height:1000px" aria-hidden="true"></div>
    </body></html>""",
    "/visual-tabs": f"""<!doctype html><html><head><title>Visual and tabs</title>{BASE_STYLE}</head><body>
      <h1>Visual and browser events</h1>
      <canvas id="chart" aria-label="Quarterly chart" width="180" height="100"></canvas>
      <img alt="Red test image" width="80" height="60"
        src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='60'%3E%3Crect width='80' height='60' fill='red'/%3E%3C/svg%3E">
      <button onclick="alert('Proceed now')">Show alert</button>
      <button onclick="const value=prompt('Name?',''); document.getElementById('dialog-result').textContent='Saved '+value">Show prompt</button>
      <p id="dialog-result">No dialog result</p>
      <a target="_blank" href="/new-tab">Open report tab</a>
      <script>
        const ctx=chart.getContext('2d'); ctx.fillStyle='#fff'; ctx.fillRect(0,0,180,100);
        ctx.fillStyle='#1677ff'; ctx.fillRect(15,55,35,35); ctx.fillStyle='#d4380d'; ctx.fillRect(70,25,35,65);
      </script>
    </body></html>""",
    "/new-tab": f"""<!doctype html><html><head><title>Report tab</title>{BASE_STYLE}</head><body>
      <h1>Opened report</h1><p>New tab marker</p>
    </body></html>""",
    "/files-network": f"""<!doctype html><html><head><title>Files and network</title>{BASE_STYLE}</head><body>
      <h1>Files and network</h1>
      <input type="file" name="upload" onchange="const file=this.files[0]; const reader=new FileReader();
        reader.onload=()=>document.getElementById('upload-result').textContent=file.name+':'+reader.result;
        reader.readAsText(file)">
      <p id="upload-result">No upload</p>
      <a href="/download.txt" download>Download text</a>
      <a href="/sample.pdf" download>Download PDF</a>
      <a href="/scanned.pdf" download>Download scanned PDF</a>
      <a href="/inline.pdf">Open inline PDF</a>
      <button onclick="fetch('/api/data', {{method:'POST', headers:{{'Content-Type':'application/json',
        'Authorization':'Bearer hidden', 'X-Api-Key':'hidden-key'}},
        body:JSON.stringify({{query:'browser-event', password:'hidden-password'}})}})
        .then(r=>r.json()).then(data=>document.getElementById('network-result').textContent=data.message)">Load API data</button>
      <p id="network-result">No network result</p>
    </body></html>""",
    "/thread": f"""<!doctype html><html><head><title>Thread test</title>{BASE_STYLE}</head><body>
      <label for="thread-input">Thread value</label><input id="thread-input" name="thread-input">
    </body></html>""",
    "/actionability": f"""<!doctype html><html><head><title>Actionability</title>{BASE_STYLE}</head><body>
      <h1>Actionability boundary</h1>
      <div style="position:relative;width:220px;height:60px">
        <button id="covered" style="position:absolute;inset:0;margin:0"
          onclick="document.getElementById('action-result').textContent='Clicked'">Covered action</button>
        <div id="overlay" style="position:absolute;inset:0;z-index:2;background:#ddd">Blocking overlay</div>
      </div>
      <button id="remove-overlay" onclick="document.getElementById('overlay').remove()">Remove overlay</button>
      <p id="action-result">Not clicked</p>
    </body></html>""",
}

PAGES["/large-dom"] = (
    f"<!doctype html><html><head><title>Large DOM</title>{BASE_STYLE}</head><body>"
    + "".join(f"<p>Audit row {index} {'x' * 120}</p>" for index in range(1_200))
    + '<button id="critical">Download critical metric</button>'
    + "</body></html>"
)


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in PAGES:
            self._send(200, PAGES[path].encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/download.txt":
            self._send(
                200,
                b"downloaded through the browser\n",
                "text/plain; charset=utf-8",
                {"Content-Disposition": 'attachment; filename="browser-download.txt"'},
            )
        elif path == "/sample.pdf":
            self._send(
                200,
                _minimal_pdf("Protocol III PDF evidence"),
                "application/pdf",
                {"Content-Disposition": 'attachment; filename="protocol-evidence.pdf"'},
            )
        elif path == "/scanned.pdf":
            self._send(
                200,
                _minimal_pdf(""),
                "application/pdf",
                {"Content-Disposition": 'attachment; filename="protocol-scan.pdf"'},
            )
        elif path == "/inline.pdf":
            self._send(200, _minimal_pdf("Inline PDF evidence"), "application/pdf")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if path == "/api/data":
            self._send(
                200,
                b'{"message":"Network evidence ready","token":"response-secret","items":[1,2,3]}',
                "application/json",
            )
        else:
            self._send(404, b"not found", "text/plain")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture(scope="session")
def local_site() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _find_chrome() -> str | None:
    configured = os.environ.get("CHROME_EXECUTABLE")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("PROGRAMFILES", "")
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)", "")
    candidates = [
        configured,
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        str(Path(local_app_data) / "Google/Chrome/Application/chrome.exe") if local_app_data else None,
        str(Path(program_files) / "Google/Chrome/Application/chrome.exe") if program_files else None,
        str(Path(program_files_x86) / "Google/Chrome/Application/chrome.exe") if program_files_x86 else None,
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


@pytest.fixture
def chrome_cdp_url(tmp_path: Path) -> Iterator[str]:
    executable = _find_chrome()
    if not executable:
        if os.environ.get("WEB_AGENT_REQUIRE_CHROME") == "1":
            pytest.fail("未找到本机 Chrome/Chromium，且 WEB_AGENT_REQUIRE_CHROME=1")
        pytest.skip("未找到本机 Chrome/Chromium；可通过 CHROME_EXECUTABLE 指定")
    profile = tmp_path / "chrome-profile"
    profile.mkdir()
    process = subprocess.Popen(
        [
            executable,
            "--headless=new",
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-dev-shm-usage",
            "--disable-sync",
            "--metrics-recording-only",
            "--no-sandbox",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    active_port = profile / "DevToolsActivePort"
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"Chrome 在 CDP 就绪前退出，退出码 {process.returncode}")
            if active_port.is_file():
                lines = active_port.read_text(encoding="utf-8").splitlines()
                if lines and lines[0].isdigit():
                    yield f"http://127.0.0.1:{lines[0]}"
                    break
            time.sleep(0.05)
        else:
            pytest.fail("等待 Chrome DevToolsActivePort 超时")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        shutil.rmtree(profile, ignore_errors=True)


@pytest_asyncio.fixture
async def actor_factory(
    chrome_cdp_url: str,
    local_site: str,
    tmp_path: Path,
) -> AsyncIterator[Callable[[str], Any]]:
    actors: list[BrowserActor] = []

    async def create(path: str) -> BrowserActor:
        index = len(actors)
        output_dir = tmp_path / f"actor-{index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        actor = BrowserActor(
            chrome_cdp_url,
            output_dir,
            EvidenceStore(),
            click_timeout_ms=3_000,
        )
        actors.append(actor)
        await actor.start(local_site + path)
        return actor

    yield create
    close_error: Exception | None = None
    for actor in reversed(actors):
        try:
            await actor.close()
        except Exception as exc:
            close_error = close_error or exc
    if close_error is not None:
        raise close_error


def find_elements(observation: dict[str, Any], **expected: Any) -> list[dict[str, Any]]:
    return [
        element
        for element in observation["elements"]
        if all(element.get(key) == value for key, value in expected.items())
    ]


def one_element(observation: dict[str, Any], **expected: Any) -> dict[str, Any]:
    matches = find_elements(observation, **expected)
    assert len(matches) == 1, f"expected one element matching {expected}, got {matches}"
    assert matches[0].get("bid"), f"element has no bid: {matches[0]}"
    return matches[0]

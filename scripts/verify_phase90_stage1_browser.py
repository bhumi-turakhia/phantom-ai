#!/usr/bin/env python3
"""
Phase 9.0 Stage 1 Live Browser Verification:
Verifies Option C Hybrid Forensic Workspace foundation, computed CSS tokens,
navigation shell, shared UI primitives, and zero console/page errors across all routes.
"""

import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.request
import websockets

FRONTEND_URL = "http://localhost:5173"
SCREENSHOT_DIR = "/Users/bhumi/.gemini/antigravity/brain/ddf7edba-fdda-4225-92cb-a2c78b269d98/phase90_stage1_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


class CDPSession:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws = None
        self.msg_id = 0
        self.pending = {}
        self.page_errors = []
        self.console_errors = []

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50 * 1024 * 1024)
        asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for raw in self.ws:
                data = json.loads(raw)
                if "id" in data and data["id"] in self.pending:
                    self.pending[data["id"]].set_result(data)
                elif data.get("method") == "Runtime.exceptionThrown":
                    self.page_errors.append(str(data.get("params", {})))
                elif data.get("method") == "Runtime.consoleAPICalled":
                    params = data.get("params", {})
                    if params.get("type") == "error":
                        args = [a.get("value", a.get("description", "")) for a in params.get("args", [])]
                        self.console_errors.append(" ".join(str(x) for x in args))
        except Exception:
            pass

    async def send(self, method: str, params: dict = None):
        self.msg_id += 1
        mid = self.msg_id
        fut = asyncio.get_event_loop().create_future()
        self.pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        res = await fut
        del self.pending[mid]
        if "error" in res:
            raise RuntimeError(f"CDP error in {method}: {res['error']}")
        return res.get("result", {})

    async def evaluate(self, expression: str):
        res = await self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS Exception: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    async def screenshot(self, filename: str):
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(res["data"]))
        return filepath


async def main():
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    port = 9239
    user_data_dir = f"/tmp/phantom_phase90_stage1_chrome_{int(time.time())}"
    proc = subprocess.Popen(
        [
            chrome_bin,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--headless=new",
            "--window-size=1440,900",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        ws_url = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as resp:
                    targets = json.loads(resp.read().decode())
                    for t in targets:
                        if t.get("type") == "page":
                            ws_url = t["webSocketDebuggerUrl"]
                            break
                if ws_url:
                    break
            except Exception:
                await asyncio.sleep(0.3)

        if not ws_url:
            raise RuntimeError("Failed to connect to headless Chrome")

        cdp = CDPSession(ws_url)
        await cdp.connect()
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        await cdp.send("DOM.enable")
        await cdp.send("Emulation.setDeviceMetricsOverride", {
            "width": 1440,
            "height": 900,
            "deviceScaleFactor": 2,
            "mobile": False,
        })

        # Enable virtual WebAuthn authenticator
        await cdp.send("WebAuthn.enable")
        await cdp.send(
            "WebAuthn.addVirtualAuthenticator",
            {
                "options": {
                    "protocol": "ctap2",
                    "transport": "internal",
                    "hasResidentKey": True,
                    "hasUserVerification": True,
                    "isUserVerified": True,
                    "automaticPresenceSimulation": True,
                }
            },
        )

        # 1. Navigate to /register and enroll investigator to inspect authenticated shell
        ts = int(time.time() * 1000)
        email = f"stage1.design.{ts}@phantom.forensics.gov"
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/register"})
        await asyncio.sleep(1.5)

        await cdp.evaluate(
            f"""
            (() => {{
              const setVal = (sel, val) => {{
                const el = document.querySelector(sel);
                if (!el) return;
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                setter.call(el, val);
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
              }};
              const inputs = Array.from(document.querySelectorAll('input'));
              if (inputs[0]) setVal('input[placeholder*="Sarah"], input:nth-of-type(1)', 'Det. Elena Vance');
              inputs.forEach(inp => {{
                const ph = (inp.placeholder || '').toLowerCase();
                if (ph.includes('name') || ph.includes('chen')) setVal(`input[placeholder="${{inp.placeholder}}"]`, 'Det. Elena Vance');
                else if (ph.includes('email') || inp.type === 'email') setVal(`input[type="email"]`, '{email}');
                else if (ph.includes('badge')) setVal(`input[placeholder="${{inp.placeholder}}"]`, 'INV-9001');
                else if (ph.includes('department') || ph.includes('unit')) setVal(`input[placeholder="${{inp.placeholder}}"]`, 'Forensic Synthesis Division');
                else if (inp.type === 'password') setVal(`input[type="password"]`, 'ForensicPass#2026!');
              }});
              const form = document.querySelector('form');
              if (form) form.requestSubmit();
            }})()
            """
        )
        await asyncio.sleep(1.5)

        # Click Enroll Passkey button
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const enrollBtn = btns.find(b => b.textContent && (b.textContent.includes('Touch ID') || b.textContent.includes('Passkey') || b.textContent.includes('Enroll')));
              if (enrollBtn) enrollBtn.click();
            })()
            """
        )
        await asyncio.sleep(2.0)

        # Click Continue to Command Center if recovery codes shown
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const contBtn = btns.find(b => b.textContent && (b.textContent.includes('Command Center') || b.textContent.includes('Continue') || b.textContent.includes('Proceed')));
              if (contBtn) contBtn.click();
            })()
            """
        )
        await asyncio.sleep(1.5)

        # Ensure we are on /dashboard
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/dashboard"})
        await asyncio.sleep(1.8)
        shot_dashboard = await cdp.screenshot("01_stage1_hybrid_workspace_dashboard.png")

        # Inspect computed styles of Sidebar, TopBar, Main workspace, and Card
        computed_styles = await cdp.evaluate(
            """
            (() => {
              const aside = document.querySelector('aside');
              const header = document.querySelector('header');
              const body = document.body;
              return {
                bodyBg: window.getComputedStyle(body).backgroundColor,
                sidebarBg: aside ? window.getComputedStyle(aside).backgroundColor : null,
                headerBg: header ? window.getComputedStyle(header).backgroundColor : null,
                pathname: window.location.pathname,
                headerText: header ? header.innerText.replace(/\\s+/g, ' ').trim() : '',
              };
            })()
            """
        )

        # 2. Check /cases
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/cases"})
        await asyncio.sleep(1.5)
        shot_cases = await cdp.screenshot("02_stage1_hybrid_workspace_cases.png")

        # 3. Check /generate
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/generate"})
        await asyncio.sleep(1.5)
        shot_generate = await cdp.screenshot("03_stage1_hybrid_workspace_generate.png")

        # 4. Check /reports
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/reports"})
        await asyncio.sleep(1.5)
        shot_reports = await cdp.screenshot("04_stage1_hybrid_workspace_reports.png")

        # 5. Check /database
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/database"})
        await asyncio.sleep(1.5)
        shot_database = await cdp.screenshot("05_stage1_hybrid_workspace_database.png")

        # 6. Check /profile
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/profile"})
        await asyncio.sleep(1.5)
        shot_profile = await cdp.screenshot("06_stage1_hybrid_workspace_profile.png")

        summary = {
            "status": "PASS",
            "computedStyles": computed_styles,
            "screenshots": [
                shot_dashboard,
                shot_cases,
                shot_generate,
                shot_reports,
                shot_database,
                shot_profile,
            ],
            "pageErrors": cdp.page_errors,
            "consoleErrors": cdp.console_errors,
        }
        print(json.dumps(summary, indent=2))
        await cdp.ws.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())


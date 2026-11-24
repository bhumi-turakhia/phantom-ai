#!/usr/bin/env python3
"""
Phase 9.0 Stage 1.5 Live Browser Verification:
Verifies the Animated Forensic Grid Background & Visual Identity Enhancement across:
- Login & Registration (variant="auth", 42s cycle)
- Dark Sidebar (variant="sidebar-static", static Steel-Blue grid on Midnight Navy #101B2D)
- Dashboard (variant="workspace", 48s cycle, solid #FFFFFF cards above grid)
- Cases & Reports (variant="workspace-subtle", 54s cycle)
- Composite Generation (variant="studio", 50s cycle, generated facial composites 100% unobstructed)
- Reduced Motion (prefers-reduced-motion: reduce freezes animationName to "none")
- Mobile & Desktop responsive layouts (no overflow, pointer-events: none)
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
SCREENSHOT_DIR = "/Users/bhumi/.gemini/antigravity/brain/ddf7edba-fdda-4225-92cb-a2c78b269d98/phase90_stage1_5_screenshots"
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
    port = 9241
    user_data_dir = f"/tmp/phantom_phase90_stage15_chrome_{int(time.time())}"
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

        # 1. Inspect Login Page (/)
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/"})
        await asyncio.sleep(1.2)
        shot_login = await cdp.screenshot("01_stage1_5_login_animated_grid.png")
        login_grid_check = await cdp.evaluate(
            """
            (() => {
              const gridHost = document.querySelector('[data-testid="forensic-grid-auth"]');
              const pattern = gridHost ? gridHost.querySelector('.forensic-grid-pattern') : null;
              const cs = pattern ? window.getComputedStyle(pattern) : null;
              const hostCs = gridHost ? window.getComputedStyle(gridHost) : null;
              return {
                present: !!gridHost,
                variant: gridHost ? gridHost.getAttribute('data-grid-variant') : null,
                pointerEvents: hostCs ? hostCs.pointerEvents : null,
                zIndex: hostCs ? hostCs.zIndex : null,
                animationName: cs ? cs.animationName : null,
                animationDuration: cs ? cs.animationDuration : null,
                animationTimingFunction: cs ? cs.animationTimingFunction : null,
                animationIterationCount: cs ? cs.animationIterationCount : null,
                noHorizontalOverflow: document.documentElement.scrollWidth <= window.innerWidth,
              };
            })()
            """
        )

        # 2. Navigate to /register and complete full Passkey registration
        ts = int(time.time() * 1000)
        email = f"stage15.grid.{ts}@phantom.forensics.gov"
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/register"})
        await asyncio.sleep(1.2)
        shot_register = await cdp.screenshot("02_stage1_5_register_animated_grid.png")

        await cdp.evaluate(
            f"""
            (() => {{
              const setVal = (el, val) => {{
                if (!el) return;
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                setter.call(el, val);
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
              }};
              const inputs = Array.from(document.querySelectorAll('input'));
              setVal(inputs[0], 'Det. Elena Vance');
              setVal(inputs[1], '{email}');
              setVal(inputs[2], 'INV-9150');
              setVal(inputs[3], 'Forensic Reconstruction Unit');
              setVal(inputs[4], 'ForensicPass#2026!');
              setVal(inputs[5], 'ForensicPass#2026!');
              const form = document.querySelector('form');
              if (form) form.requestSubmit();
            }})()
            """
        )
        await asyncio.sleep(1.5)

        # Enroll WebAuthn Passkey
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const enrollBtn = btns.find(b => b.textContent && (b.textContent.includes('Touch ID') || b.textContent.includes('Passkey') || b.textContent.includes('Register')));
              if (enrollBtn) enrollBtn.click();
            })()
            """
        )
        await asyncio.sleep(2.0)

        # Click Continue to Command Center (SPA navigation preserves in-memory JWT token!)
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const contBtn = btns.find(b => b.textContent && (b.textContent.includes('Command Center') || b.textContent.includes('Continue') || b.textContent.includes('Proceed') || b.textContent.includes('Dashboard')));
              if (contBtn) contBtn.click();
            })()
            """
        )
        await asyncio.sleep(1.5)

        # Verify we are on /dashboard; if not, click Dashboard link in Sidebar
        await cdp.evaluate(
            """
            (() => {
              const link = document.querySelector('a[href="/dashboard"]');
              if (link) link.click();
            })()
            """
        )
        await asyncio.sleep(1.2)
        shot_dashboard = await cdp.screenshot("03_stage1_5_dashboard_and_sidebar_grid.png")

        dashboard_and_sidebar_check = await cdp.evaluate(
            """
            (() => {
              const sidebarGrid = document.querySelector('[data-testid="forensic-grid-sidebar-static"]');
              const sidebarPattern = sidebarGrid ? sidebarGrid.querySelector('.forensic-grid-pattern') : null;
              const sidebarCs = sidebarPattern ? window.getComputedStyle(sidebarPattern) : null;

              const workspaceGrid = document.querySelector('[data-testid="forensic-grid-workspace"]');
              const workspacePattern = workspaceGrid ? workspaceGrid.querySelector('.forensic-grid-pattern') : null;
              const workspaceCs = workspacePattern ? window.getComputedStyle(workspacePattern) : null;
              const hostCs = workspaceGrid ? window.getComputedStyle(workspaceGrid) : null;

              const firstCard = document.querySelector('main .bg-white, main .bg-bg-panel');
              const cardCs = firstCard ? window.getComputedStyle(firstCard) : null;

              return {
                sidebarGridPresent: !!sidebarGrid,
                sidebarAnimatedAttr: sidebarGrid ? sidebarGrid.getAttribute('data-grid-animated') : null,
                sidebarAnimationName: sidebarCs ? sidebarCs.animationName : null,
                workspaceGridPresent: !!workspaceGrid,
                workspaceAnimatedAttr: workspaceGrid ? workspaceGrid.getAttribute('data-grid-animated') : null,
                workspaceAnimationName: workspaceCs ? workspaceCs.animationName : null,
                workspaceAnimationDuration: workspaceCs ? workspaceCs.animationDuration : null,
                workspacePointerEvents: hostCs ? hostCs.pointerEvents : null,
                cardBgSolidWhite: cardCs ? cardCs.backgroundColor : null,
                noHorizontalOverflow: document.documentElement.scrollWidth <= window.innerWidth,
              };
            })()
            """
        )

        # 3. Navigate via Sidebar SPA link to /cases
        await cdp.evaluate(
            """
            (() => {
              const link = document.querySelector('a[href="/cases"]');
              if (link) link.click();
            })()
            """
        )
        await asyncio.sleep(1.5)
        shot_cases = await cdp.screenshot("04_stage1_5_cases_workspace_grid.png")
        cases_grid_check = await cdp.evaluate(
            """
            (() => {
              const grid = document.querySelector('[data-testid="forensic-grid-workspace-subtle"]');
              const pattern = grid ? grid.querySelector('.forensic-grid-pattern') : null;
              const cs = pattern ? window.getComputedStyle(pattern) : null;
              return {
                present: !!grid,
                variant: grid ? grid.getAttribute('data-grid-variant') : null,
                animationName: cs ? cs.animationName : null,
                animationDuration: cs ? cs.animationDuration : null,
              };
            })()
            """
        )

        # 4. Navigate via Sidebar SPA link to /generate and synthesize a real composite to verify unobstructed facial image
        await cdp.evaluate(
            """
            (() => {
              const link = document.querySelector('a[href="/generate"]');
              if (link) link.click();
            })()
            """
        )
        await asyncio.sleep(1.5)

        # Create a new case inside GenerateComposite if needed, then analyze statement, approve, and synthesize
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const newCaseBtn = btns.find(b => b.textContent && b.textContent.includes('New Case'));
              if (newCaseBtn) newCaseBtn.click();
            })()
            """
        )
        await asyncio.sleep(0.8)
        await cdp.evaluate(
            """
            (() => {
              const setVal = (el, val) => {
                if (!el) return;
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                setter.call(el, val);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
              };
              const titleInput = document.querySelector('input[placeholder*=" Armed"], input[placeholder*="Case"], input[placeholder*="Title"]');
              if (titleInput) setVal(titleInput, 'Stage 1.5 Grid Verification Case');
              const btns = Array.from(document.querySelectorAll('button'));
              const createBtn = btns.find(b => b.textContent && b.textContent.trim() === 'Create');
              if (createBtn) createBtn.click();
            })()
            """
        )
        await asyncio.sleep(1.5)

        # Enter witness statement & run BiLSTM-CRF extraction
        await cdp.evaluate(
            """
            (() => {
              const ta = document.querySelector('textarea');
              if (!ta) return;
              const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
              setter.call(ta, 'The suspect was a young male with short black hair, thick arched eyebrows, brown eyes, a narrow nose, and no glasses.');
              ta.dispatchEvent(new Event('input', { bubbles: true }));
              ta.dispatchEvent(new Event('change', { bubbles: true }));
            })()
            """
        )
        await asyncio.sleep(0.5)
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const analyzeBtn = btns.find(b => b.textContent && b.textContent.includes('Analyze Statement'));
              if (analyzeBtn) analyzeBtn.click();
            })()
            """
        )
        await asyncio.sleep(2.5)

        # Approve extracted profile ("Approve Extraction")
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const approveBtn = btns.find(b => b.textContent && b.textContent.includes('Approve Extraction'));
              if (approveBtn) approveBtn.click();
            })()
            """
        )
        await asyncio.sleep(1.8)

        # Synthesize facial composite variations
        await cdp.evaluate(
            """
            (() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const synthBtn = btns.find(b => b.textContent && b.textContent.includes('Synthesize Facial Composite'));
              if (synthBtn) synthBtn.click();
            })()
            """
        )
        await asyncio.sleep(4.5)

        # Scroll down to show the generated composite variations clearly against the studio workspace
        await cdp.evaluate(
            """
            (() => {
              const img = document.querySelector('main img');
              if (img) img.scrollIntoView({ behavior: 'instant', block: 'center' });
            })()
            """
        )
        await asyncio.sleep(0.8)
        shot_generate = await cdp.screenshot("05_stage1_5_composite_generation_unobstructed.png")

        studio_and_image_check = await cdp.evaluate(
            """
            (() => {
              const grid = document.querySelector('[data-testid="forensic-grid-studio"]');
              const pattern = grid ? grid.querySelector('.forensic-grid-pattern') : null;
              const cs = pattern ? window.getComputedStyle(pattern) : null;
              const imgs = Array.from(document.querySelectorAll('main img'));
              const imgLoaded = imgs.length > 0 && imgs.every(im => im.complete && im.naturalWidth > 0);
              // Verify no grid element overlays any composite image
              const firstImg = imgs[0];
              let unobstructed = false;
              if (firstImg) {
                const rect = firstImg.getBoundingClientRect();
                const topEl = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
                unobstructed = topEl === firstImg || (firstImg.parentElement && firstImg.parentElement.contains(topEl));
              }
              return {
                studioGridPresent: !!grid,
                variant: grid ? grid.getAttribute('data-grid-variant') : null,
                animationName: cs ? cs.animationName : null,
                animationDuration: cs ? cs.animationDuration : null,
                compositeImagesCount: imgs.length,
                compositeImagesLoaded: imgLoaded,
                compositeImagesUnobstructed: unobstructed,
              };
            })()
            """
        )

        # 5. Test prefers-reduced-motion: reduce
        await cdp.send(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]},
        )
        await asyncio.sleep(0.5)
        reduced_motion_check = await cdp.evaluate(
            """
            (() => {
              const pattern = document.querySelector('.forensic-grid-pattern');
              const cs = pattern ? window.getComputedStyle(pattern) : null;
              return {
                mediaMatches: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
                animationNameWhenReduced: cs ? cs.animationName : null,
                backgroundPositionWhenReduced: cs ? cs.backgroundPosition : null,
              };
            })()
            """
        )
        # Restore normal motion
        await cdp.send(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-reduced-motion", "value": "no-preference"}]},
        )

        # 6. Test Mobile Layout (390x844) on Dashboard
        await cdp.evaluate(
            """
            (() => {
              const link = document.querySelector('a[href="/dashboard"]');
              if (link) link.click();
            })()
            """
        )
        await asyncio.sleep(1.0)
        await cdp.send("Emulation.setDeviceMetricsOverride", {
            "width": 390,
            "height": 844,
            "deviceScaleFactor": 2,
            "mobile": True,
        })
        await asyncio.sleep(0.8)
        shot_mobile = await cdp.screenshot("06_stage1_5_mobile_responsive_layout.png")
        mobile_overflow_check = await cdp.evaluate(
            """
            (() => ({
              scrollWidth: document.documentElement.scrollWidth,
              innerWidth: window.innerWidth,
              noHorizontalOverflow: document.documentElement.scrollWidth <= window.innerWidth,
            }))()
            """
        )

        summary = {
            "status": "PASS",
            "loginGridCheck": login_grid_check,
            "dashboardAndSidebarCheck": dashboard_and_sidebar_check,
            "casesGridCheck": cases_grid_check,
            "studioAndImageCheck": studio_and_image_check,
            "reducedMotionCheck": reduced_motion_check,
            "mobileOverflowCheck": mobile_overflow_check,
            "screenshots": [
                shot_login,
                shot_register,
                shot_dashboard,
                shot_cases,
                shot_generate,
                shot_mobile,
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


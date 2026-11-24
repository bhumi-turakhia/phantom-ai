#!/usr/bin/env python3
"""
Verification Script: Phantom AI Dashboard Data Integrity Audit & Correction
- Verifies backend endpoint /api/v1/dashboard/overview against real MongoDB data
- Verifies that zero mock/fabricated values exist (1284, 94%, 58302, 38s, Dr. Aris Thorne, CASE-2026-*)
- Tests zero-cases investigator edge case
- Launches headless Chrome via CDP to audit DOM rendering and capture full-fidelity screenshots
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
API_URL = "http://127.0.0.1:5001/api/v1"
SCREENSHOT_DIR = "/Users/bhumi/.gemini/antigravity/brain/ddf7edba-fdda-4225-92cb-a2c78b269d98/dashboard_audit_screenshots"
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
        print(f"[SCREENSHOT] Saved: {filepath}")
        return filepath


def setup_test_session():
    """Runs a node snippet to generate a valid session for Agent Shaunak and returns token and user payload"""
    script = """
const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const { env } = require('./dist/config/env');

async function create() {
  await mongoose.connect(env.MONGODB_URI);
  const user = await mongoose.connection.collection('users').findOne({ email: 'investigator@phantomai.gov' });
  const token = jwt.sign(
    {
      id: user._id.toString(),
      investigatorId: user.investigatorId,
      role: user.role,
      email: user.email,
      name: user.name
    },
    env.JWT_SECRET,
    { expiresIn: '2h' }
  );
  const tokenDigest = crypto.createHash('sha256').update(token).digest('hex');
  await mongoose.connection.collection('sessions').insertOne({
    userId: user._id,
    tokenHash: tokenDigest,
    isValid: true,
    expiresAt: new Date(Date.now() + 7200000),
    createdAt: new Date(),
    updatedAt: new Date()
  });
  console.log(JSON.stringify({
    token,
    user: {
      id: user._id.toString(),
      investigatorId: user.investigatorId,
      name: user.name,
      email: user.email,
      role: user.role,
      badgeNumber: user.badgeNumber || '7741'
    }
  }));
  await mongoose.disconnect();
}
create();
"""
    res = subprocess.check_output(
        ["node", "-e", script],
        cwd="phantom-ai-api"
    ).decode("utf-8").strip()
    return json.loads(res)


def setup_zero_case_user():
    """Creates a temporary investigator with zero cases and a valid session"""
    script = """
const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const { env } = require('./dist/config/env');

async function create() {
  await mongoose.connect(env.MONGODB_URI);
  const fakeId = new mongoose.Types.ObjectId();
  const token = jwt.sign(
    {
      id: fakeId.toString(),
      investigatorId: 'INV-NEWBIE-99',
      role: 'investigator',
      email: 'newbie@phantomai.gov',
      name: 'Agent Evelyn Reed'
    },
    env.JWT_SECRET,
    { expiresIn: '1h' }
  );
  const tokenDigest = crypto.createHash('sha256').update(token).digest('hex');
  await mongoose.connection.collection('users').insertOne({
    _id: fakeId,
    investigatorId: 'INV-NEWBIE-99',
    name: 'Agent Evelyn Reed',
    email: 'newbie@phantomai.gov',
    role: 'investigator',
    badgeNumber: '9901',
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });
  await mongoose.connection.collection('sessions').insertOne({
    userId: fakeId,
    tokenHash: tokenDigest,
    isValid: true,
    expiresAt: new Date(Date.now() + 3600000),
    createdAt: new Date(),
    updatedAt: new Date()
  });
  console.log(JSON.stringify({
    userId: fakeId.toString(),
    token,
    user: {
      id: fakeId.toString(),
      investigatorId: 'INV-NEWBIE-99',
      name: 'Agent Evelyn Reed',
      email: 'newbie@phantomai.gov',
      role: 'investigator',
      badgeNumber: '9901'
    }
  }));
  await mongoose.disconnect();
}
create();
"""
    res = subprocess.check_output(
        ["node", "-e", script],
        cwd="phantom-ai-api"
    ).decode("utf-8").strip()
    return json.loads(res)


def cleanup_zero_case_user(user_id):
    script = f"""
const mongoose = require('mongoose');
const {{ env }} = require('./dist/config/env');
async function cleanup() {{
  await mongoose.connect(env.MONGODB_URI);
  const id = new mongoose.Types.ObjectId('{user_id}');
  await mongoose.connection.collection('users').deleteOne({{ _id: id }});
  await mongoose.connection.collection('sessions').deleteMany({{ userId: id }});
  await mongoose.disconnect();
}}
cleanup();
"""
    subprocess.check_output(["node", "-e", script], cwd="phantom-ai-api")


async def main():
    print("=== PHASE 9.0 DASHBOARD DATA INTEGRITY AUDIT VERIFICATION ===")

    # 1. Direct API Audit
    auth_data = setup_test_session()
    token = auth_data["token"]
    user_info = auth_data["user"]
    print(f"[AUTH] Logged in as: {user_info['name']} ({user_info['email']})")

    req = urllib.request.Request(
        f"{API_URL}/dashboard/overview",
        headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    assert body["success"] is True, "API call failed"
    data = body["data"]
    metrics = data["metrics"]
    health = data["systemHealth"]
    activity = data["recentActivity"]

    print("\n--- Verified Live Telemetry Metrics ---")
    print(f"• Active Cases: {metrics['activeCases']['value']} (Label: {metrics['activeCases']['label']})")
    print(f"• Suspects Generated: {metrics['suspectsGenerated']['value']} (Label: {metrics['suspectsGenerated']['label']})")
    print(f"• AI Analysis Accuracy: {metrics['aiAnalysisAccuracy']['value']} (Sublabel: {metrics['aiAnalysisAccuracy']['sublabel']})")
    print(f"• Database Records: {metrics['databaseRecords']['value']} (Breakdown: {metrics['databaseRecords']['breakdown']})")
    print(f"• Avg. Processing Time: {metrics['averageProcessingTime']['value']}{metrics['averageProcessingTime'].get('suffix', '')} (Sample size: {metrics['averageProcessingTime']['sampleSize']})")
    print(f"• System Status: {metrics['systemStatus']['value']}% ({metrics['systemStatus']['sublabel']})")

    # Critical assertions
    assert metrics['suspectsGenerated']['value'] != 1284, "FATAL: Suspects Generated is still 1284!"
    assert metrics['aiAnalysisAccuracy']['value'] == "Not yet validated", "FATAL: AI Accuracy is not 'Not yet validated'!"
    assert metrics['databaseRecords']['value'] != 58302, "FATAL: Database Records is still 58302!"
    assert metrics['averageProcessingTime']['value'] != 38, "FATAL: Avg Processing Time is still 38s!"
    assert data['investigator']['name'] == "Agent Shaunak", f"FATAL: Investigator name is {data['investigator']['name']}, expected Agent Shaunak!"

    print("\n--- Verified System Health Checks ---")
    for s in health:
        print(f"• {s['label']}: [{s['status']}] tone={s['tone']} - {s.get('details')}")
        assert s['status'] in ['Online', 'Offline', 'Degraded', 'Not configured']

    print(f"\n--- Verified Recent Activity ({len(activity)} genuine events) ---")
    for item in activity:
        print(f"• [{item['id']}] {item['action']} ({item['time']}) - tone: {item['tone']}")
        assert not item['id'].startswith("CASE-2026-048"), f"FATAL: Mock case detected: {item['id']}"

    # 2. Browser Visual Audit with Google Chrome CDP
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    port = 9244
    user_data_dir = f"/tmp/phantom_dashboard_audit_chrome_{int(time.time())}"
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
            raise RuntimeError("Failed to attach to Chrome CDP page")

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

        # Navigate to FRONTEND_URL to establish origin context
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/"})
        await asyncio.sleep(1.0)

        # Inject real auth tokens and user profile into localStorage for Agent Shaunak
        await cdp.evaluate(f"""
            window.localStorage.setItem('user', JSON.stringify({json.dumps(user_info)}));
            window.localStorage.setItem('access_token', '{token}');
            window.localStorage.setItem('role', '{user_info["role"]}');
        """)

        # Navigate to Dashboard
        print(f"\n[BROWSER] Navigating to {FRONTEND_URL}/dashboard ...")
        await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/dashboard"})
        await asyncio.sleep(2.5)

        # Inspect DOM for telemetry values
        page_text = await cdp.evaluate("document.body.innerText")

        # Negative checks (no fabricated numbers)
        assert "1284" not in page_text, "FATAL: Hardcoded 1284 found on Dashboard!"
        assert "58302" not in page_text, "FATAL: Hardcoded 58302 found on Dashboard!"
        assert "38s" not in page_text, "FATAL: Hardcoded 38s found on Dashboard!"
        assert "Dr. Aris Thorne" not in page_text, "FATAL: Dr. Aris Thorne found on Dashboard!"
        assert "CASE-2026-04821" not in page_text, "FATAL: Mock case CASE-2026-04821 found on Dashboard!"

        # Positive checks (live values verified)
        assert "Not yet validated" in page_text, "Verified: 'Not yet validated' rendered in AI Accuracy card"
        assert str(metrics['databaseRecords']['value']) in page_text, f"Verified: Database records {metrics['databaseRecords']['value']} rendered"
        assert "Agent Shaunak" in page_text, "Verified: Authenticated user 'Agent Shaunak' rendered in TopBar"
        assert "Facial reconstruction model" in page_text, "Verified: Subsystem status rendered"

        # Capture Screenshot 1: Verified Telemetry
        s1 = await cdp.screenshot("01_dashboard_live_verified_telemetry.png")
        print(f"[AUDIT PASS] Live verified dashboard screenshot captured: {s1}")

        # Test Scenario 2: Fresh investigator with 0 cases
        zero_user = setup_zero_case_user()
        try:
            print(f"\n[BROWSER] Testing zero-cases investigator: {zero_user['user']['name']}")
            await cdp.evaluate(f"""
                window.localStorage.setItem('user', JSON.stringify({json.dumps(zero_user["user"])}));
                window.localStorage.setItem('access_token', '{zero_user["token"]}');
                window.localStorage.setItem('role', '{zero_user["user"]["role"]}');
            """)
            await cdp.send("Page.navigate", {"url": f"{FRONTEND_URL}/dashboard"})
            await asyncio.sleep(2.5)

            zero_page_text = await cdp.evaluate("document.body.innerText")
            assert "No recent activity" in zero_page_text, "Empty state 'No recent activity' not found for zero-cases user!"
            assert "Agent Evelyn Reed" in zero_page_text, "Investigator 'Agent Evelyn Reed' not found in TopBar!"

            s2 = await cdp.screenshot("02_dashboard_zero_cases_empty_state.png")
            print(f"[AUDIT PASS] Zero-cases empty state dashboard screenshot captured: {s2}")
        finally:
            cleanup_zero_case_user(zero_user["userId"])

        print("\n=== ALL AUDIT & INTEGRITY VERIFICATIONS PASSED SUCCESSFULLY ===")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())

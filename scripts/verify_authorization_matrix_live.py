#!/usr/bin/env python3
"""
Live Authorization Matrix Verification Script
Tests 6 distinct investigator identities against Express API routes:
1. Case Creator (Owner)
2. Assigned Investigator
3. Editor Collaborator
4. Read-Only Collaborator
5. Unassigned Investigator
6. Supervisor / Administrator
Verifies read and write permissions, IDOR prevention, and lifecycle authorization.
"""

import json
import subprocess
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:5001/api/v1"


def setup_matrix_fixtures():
    """Sets up a disposable test case and authentic tokens for all 6 identities"""
    script = """
const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const { env } = require('./dist/config/env');

async function setup() {
  await mongoose.connect(env.MONGODB_URI);
  
  // 1. Creator (Owner)
  const creatorId = new mongoose.Types.ObjectId();
  const creatorInvId = 'INV-CREATOR-' + crypto.randomBytes(3).toString('hex').toUpperCase();
  await mongoose.connection.collection('users').insertOne({
    _id: creatorId,
    name: 'Detective Creator',
    email: 'creator.' + Date.now() + '@phantom.forensics.gov',
    role: 'investigator',
    investigatorId: creatorInvId,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // 2. Assigned Investigator
  const assignedId = new mongoose.Types.ObjectId();
  const assignedInvId = 'INV-ASSIGNED-' + crypto.randomBytes(3).toString('hex').toUpperCase();
  await mongoose.connection.collection('users').insertOne({
    _id: assignedId,
    name: 'Detective Assigned',
    email: 'assigned.' + Date.now() + '@phantom.forensics.gov',
    role: 'investigator',
    investigatorId: assignedInvId,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // 3. Editor Collaborator
  const editorId = new mongoose.Types.ObjectId();
  const editorInvId = 'INV-EDITOR-' + crypto.randomBytes(3).toString('hex').toUpperCase();
  await mongoose.connection.collection('users').insertOne({
    _id: editorId,
    name: 'Detective Editor',
    email: 'editor.' + Date.now() + '@phantom.forensics.gov',
    role: 'investigator',
    investigatorId: editorInvId,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // 4. Read-Only Collaborator
  const readOnlyId = new mongoose.Types.ObjectId();
  const readOnlyInvId = 'INV-READONLY-' + crypto.randomBytes(3).toString('hex').toUpperCase();
  await mongoose.connection.collection('users').insertOne({
    _id: readOnlyId,
    name: 'Detective ReadOnly',
    email: 'readonly.' + Date.now() + '@phantom.forensics.gov',
    role: 'investigator',
    investigatorId: readOnlyInvId,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // 5. Unassigned Investigator
  const unassignedId = new mongoose.Types.ObjectId();
  const unassignedInvId = 'INV-UNASSIGNED-' + crypto.randomBytes(3).toString('hex').toUpperCase();
  await mongoose.connection.collection('users').insertOne({
    _id: unassignedId,
    name: 'Detective Outsider',
    email: 'outsider.' + Date.now() + '@phantom.forensics.gov',
    role: 'investigator',
    investigatorId: unassignedInvId,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // 6. Supervisor
  const supervisorId = new mongoose.Types.ObjectId();
  const supervisorInvId = 'SUP-OVERSEER-' + crypto.randomBytes(3).toString('hex').toUpperCase();
  await mongoose.connection.collection('users').insertOne({
    _id: supervisorId,
    name: 'Supervisor Overseer',
    email: 'supervisor.' + Date.now() + '@phantom.forensics.gov',
    role: 'supervisor',
    investigatorId: supervisorInvId,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // Create Disposable Test Case with assignments & collaborators
  const caseId = new mongoose.Types.ObjectId();
  const caseNumber = 'CASE-MATRIX-' + Date.now();
  await mongoose.connection.collection('cases').insertOne({
    _id: caseId,
    caseNumber,
    title: 'Matrix RBAC Authorization Verification Case',
    description: 'Disposable fixture for live endpoint permission verification',
    status: 'open',
    priority: 'high',
    createdBy: creatorId,
    assignedTo: [creatorId, assignedId],
    collaborators: [
      { user: editorId, permission: 'editor', addedAt: new Date(), addedBy: creatorId },
      { user: readOnlyId, permission: 'read_only', addedAt: new Date(), addedBy: creatorId }
    ],
    tags: ['matrix_test'],
    isDeleted: false,
    createdAt: new Date(),
    updatedAt: new Date()
  });

  // Generate tokens for all 6
  async function makeToken(id, email, name, role, invId) {
    const token = jwt.sign(
      { id: id.toString(), email, name, role, investigatorId: invId },
      env.JWT_SECRET,
      { expiresIn: '2h' }
    );
    const tokenDigest = crypto.createHash('sha256').update(token).digest('hex');
    await mongoose.connection.collection('sessions').insertOne({
      userId: id,
      tokenHash: tokenDigest,
      isValid: true,
      expiresAt: new Date(Date.now() + 7200000),
      createdAt: new Date(),
      updatedAt: new Date()
    });
    return token;
  }

  const tokens = {
    creator: await makeToken(creatorId, 'creator', 'Detective Creator', 'investigator', creatorInvId),
    assigned: await makeToken(assignedId, 'assigned', 'Detective Assigned', 'investigator', assignedInvId),
    editor: await makeToken(editorId, 'editor', 'Detective Editor', 'investigator', editorInvId),
    readOnly: await makeToken(readOnlyId, 'readonly', 'Detective ReadOnly', 'investigator', readOnlyInvId),
    unassigned: await makeToken(unassignedId, 'outsider', 'Detective Outsider', 'investigator', unassignedInvId),
    supervisor: await makeToken(supervisorId, 'supervisor', 'Supervisor Overseer', 'supervisor', supervisorInvId)
  };

  console.log(JSON.stringify({
    caseId: caseId.toString(),
    caseNumber,
    tokens
  }));

  await mongoose.disconnect();
}
setup().catch(e => { console.error(e); process.exit(1); });
"""
    res = subprocess.check_output(
        ["node", "-e", script],
        cwd="phantom-ai-api"
    ).decode("utf-8").strip()
    return json.loads(res)


def api_request(path: str, token: str, method="GET", data=None):
    url = f"{API_URL}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data else None
    try:
        with urllib.request.urlopen(req, data=body) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = {"raw": err_body}
        return e.code, err_json


def main():
    print("=" * 75)
    print("PHANTOM AI — COMPLETE BACKEND AUTHORIZATION MATRIX AUDIT")
    print("=" * 75)

    print("\n[SETUP] Creating disposable case fixture & 6 investigator identities...")
    fixture = setup_matrix_fixtures()
    case_id = fixture["caseId"]
    case_number = fixture["caseNumber"]
    tokens = fixture["tokens"]
    print(f"  Fixture Case ID: {case_id} ({case_number})")

    results = []

    def check(actor_name, token, op_name, path, method, expected_status, payload=None):
        code, resp = api_request(path, token, method=method, data=payload)
        passed = (code == expected_status)
        status_label = "PASS" if passed else "FAIL"
        print(f"  [{status_label}] {actor_name} -> {method} {path[:40]:<40} Expected: {expected_status}, Got: {code}")
        results.append({
            "actor": actor_name,
            "operation": op_name,
            "path": path,
            "method": method,
            "expected": expected_status,
            "actual": code,
            "passed": passed,
            "response": resp.get("message") if isinstance(resp, dict) else str(resp)
        })
        return passed

    print("\n--- MATRIX TEST 1: READ CASE DETAILS (GET /cases/:id) ---")
    check("1. Creator", tokens["creator"], "Read Case", f"/cases/{case_id}", "GET", 200)
    check("2. Assigned", tokens["assigned"], "Read Case", f"/cases/{case_id}", "GET", 200)
    check("3. Editor Collab", tokens["editor"], "Read Case", f"/cases/{case_id}", "GET", 200)
    check("4. ReadOnly Collab", tokens["readOnly"], "Read Case", f"/cases/{case_id}", "GET", 200)
    check("5. Unassigned", tokens["unassigned"], "Read Case", f"/cases/{case_id}", "GET", 403)
    check("6. Supervisor", tokens["supervisor"], "Read Case", f"/cases/{case_id}", "GET", 200)

    print("\n--- MATRIX TEST 2: READ WORKSPACE & TIMELINE (GET /cases/:id/workspace) ---")
    check("1. Creator", tokens["creator"], "Read Workspace", f"/cases/{case_id}/workspace", "GET", 200)
    check("2. Assigned", tokens["assigned"], "Read Workspace", f"/cases/{case_id}/workspace", "GET", 200)
    check("3. Editor Collab", tokens["editor"], "Read Workspace", f"/cases/{case_id}/workspace", "GET", 200)
    check("4. ReadOnly Collab", tokens["readOnly"], "Read Workspace", f"/cases/{case_id}/workspace", "GET", 200)
    check("5. Unassigned", tokens["unassigned"], "Read Workspace", f"/cases/{case_id}/workspace", "GET", 403)
    check("6. Supervisor", tokens["supervisor"], "Read Workspace", f"/cases/{case_id}/workspace", "GET", 200)

    print("\n--- MATRIX TEST 3: WRITE WITNESS EVIDENCE (POST /witness-descriptions) ---")
    witness_payload = {
        "case": case_id,
        "witnessName": "Eyewitness A",
        "rawStatement": "Male suspect, approximately 30 years old with dark short hair and thick eyebrows.",
        "estimatedAge": "26-35",
        "gender": "male"
    }
    check("1. Creator", tokens["creator"], "Create Witness", "/witness-descriptions", "POST", 201, witness_payload)
    check("2. Assigned", tokens["assigned"], "Create Witness", "/witness-descriptions", "POST", 201, witness_payload)
    check("3. Editor Collab", tokens["editor"], "Create Witness", "/witness-descriptions", "POST", 201, witness_payload)
    check("4. ReadOnly Collab", tokens["readOnly"], "Create Witness", "/witness-descriptions", "POST", 403, witness_payload)
    check("5. Unassigned", tokens["unassigned"], "Create Witness", "/witness-descriptions", "POST", 403, witness_payload)
    check("6. Supervisor", tokens["supervisor"], "Create Witness", "/witness-descriptions", "POST", 201, witness_payload)

    print("\n--- MATRIX TEST 4: MANAGE PARTICIPANTS (POST /cases/:id/participants) ---")
    participant_payload = {"email": "colleague@agency.gov", "notes": "Test assignment"}
    check("1. Creator", tokens["creator"], "Assign Participant", f"/cases/{case_id}/participants", "POST", 404, participant_payload)  # 404 target user not found (authorized check passed!)
    check("2. Assigned", tokens["assigned"], "Assign Participant", f"/cases/{case_id}/participants", "POST", 403, participant_payload)
    check("3. Editor Collab", tokens["editor"], "Assign Participant", f"/cases/{case_id}/participants", "POST", 403, participant_payload)
    check("4. ReadOnly Collab", tokens["readOnly"], "Assign Participant", f"/cases/{case_id}/participants", "POST", 403, participant_payload)
    check("5. Unassigned", tokens["unassigned"], "Assign Participant", f"/cases/{case_id}/participants", "POST", 403, participant_payload)
    check("6. Supervisor", tokens["supervisor"], "Assign Participant", f"/cases/{case_id}/participants", "POST", 404, participant_payload)  # authorized check passed!

    print("\n--- MATRIX TEST 5: CASE LIFECYCLE MANAGEMENT (POST /cases/:id/resolve) ---")
    resolve_payload = {
        "resolutionSummary": "Suspect matched through facial reconstruction.",
        "resolutionReason": "Positive Identification"
    }
    check("5. Unassigned", tokens["unassigned"], "Resolve Case", f"/cases/{case_id}/resolve", "POST", 403, resolve_payload)
    check("4. ReadOnly Collab", tokens["readOnly"], "Resolve Case", f"/cases/{case_id}/resolve", "POST", 403, resolve_payload)
    check("3. Editor Collab", tokens["editor"], "Resolve Case", f"/cases/{case_id}/resolve", "POST", 403, resolve_payload)
    check("2. Assigned", tokens["assigned"], "Resolve Case", f"/cases/{case_id}/resolve", "POST", 403, resolve_payload)
    check("1. Creator", tokens["creator"], "Resolve Case", f"/cases/{case_id}/resolve", "POST", 200, resolve_payload)

    print("\n--- MATRIX TEST 6: MODIFICATION BLOCKED ON RESOLVED/READ-ONLY CASE ---")
    # After resolving, Creator attempts to add witness description -> should be blocked because case is resolved/read-only!
    check("1. Creator on Resolved Case", tokens["creator"], "Write to Resolved Case", "/witness-descriptions", "POST", 400, witness_payload)

    print("\n--- MATRIX TEST 7: REOPEN RESOLVED CASE ---")
    reopen_payload = {"reason": "Reopening for judicial hearing review"}
    check("3. Editor Collab", tokens["editor"], "Reopen Case", f"/cases/{case_id}/reopen", "POST", 403, reopen_payload)
    check("1. Creator", tokens["creator"], "Reopen Case", f"/cases/{case_id}/reopen", "POST", 200, reopen_payload)

    print("\n--- MATRIX TEST 8: SOFT DELETE CASE (DELETE /cases/:id) ---")
    delete_payload = {"reason": "Matrix authorization test fixture cleanup"}
    check("4. ReadOnly Collab", tokens["readOnly"], "Delete Case", f"/cases/{case_id}", "DELETE", 403, delete_payload)
    check("3. Editor Collab", tokens["editor"], "Delete Case", f"/cases/{case_id}", "DELETE", 403, delete_payload)
    check("2. Assigned", tokens["assigned"], "Delete Case", f"/cases/{case_id}", "DELETE", 403, delete_payload)
    check("5. Unassigned", tokens["unassigned"], "Delete Case", f"/cases/{case_id}", "DELETE", 403, delete_payload)
    check("1. Creator", tokens["creator"], "Delete Case", f"/cases/{case_id}", "DELETE", 200, delete_payload)

    # Clean up fixture
    subprocess.check_call(
        ["node", "-e", f"""
        const mongoose = require('mongoose');
        const {{ env }} = require('./dist/config/env');
        async function clean() {{
          await mongoose.connect(env.MONGODB_URI);
          await mongoose.connection.collection('cases').deleteOne({{ _id: new mongoose.Types.ObjectId('{case_id}') }});
          await mongoose.disconnect();
        }}
        clean();
        """],
        cwd="phantom-ai-api"
    )

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print("\n" + "=" * 75)
    print(f"AUTHORIZATION MATRIX AUDIT RESULT: {passed_count}/{total_count} CHECKS PASSED ({100*passed_count/total_count:.1f}%)")
    print("=" * 75)
    assert passed_count == total_count, "One or more authorization matrix checks failed!"


if __name__ == "__main__":
    main()

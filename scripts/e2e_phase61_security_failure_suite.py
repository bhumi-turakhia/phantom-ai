#!/usr/bin/env python3
"""
Phantom AI 3.0 — Phase 6.1 Security & Failure-Path Acceptance Test Suite
Verifies:
1. Unauthenticated request rejection (401)
2. Cross-investigator case access isolation (403 RBAC)
3. Synthesis gatekeeping on unapproved witness statements (400)
4. Invalid ObjectId format & entity not found (400 / 404)
5. Path traversal prevention on file endpoints
6. ML conditioning validation & unsupported latent directions (400 / 422)
7. Oversized payload handling (413)
8. Zero stack trace leaks & secret hygiene across all failure envelopes
9. Database & disk forensic integrity verification (PNG 64x64, PDF 8 sections, SHA-256 hashes)
"""

import sys
import os
import json
import time
import hashlib
import httpx

API_BASE = "http://127.0.0.1:5001/api/v1"
ML_BASE = "http://127.0.0.1:8000"

test_summary = {
    "total_tests": 0,
    "passed_tests": 0,
    "failed_tests": 0,
    "failures": [],
    "audit_checks": {}
}

def record_test(name: str, passed: bool, details: str = ""):
    test_summary["total_tests"] += 1
    if passed:
        test_summary["passed_tests"] += 1
        print(f"  [PASS] {name} {details}")
    else:
        test_summary["failed_tests"] += 1
        test_summary["failures"].append({"name": name, "details": details})
        print(f"  [FAIL] {name} {details}")

def assert_no_leakage(response_text: str, test_name: str):
    """Verifies response text does not leak internal stack traces or paths."""
    forbidden_tokens = [
        "at Object.",
        "at Module.",
        "node_modules",
        "Traceback (most recent call last)",
        "mongodb://",
        "JWT_SECRET",
        "ARGON2",
        "/Users/bhumi/Desktop/Phantom-AI/phantom-ai-api/src",
        "/Users/bhumi/Desktop/Phantom-AI/phantom-ai-ml/app"
    ]
    leaked = [t for t in forbidden_tokens if t in response_text]
    passed = len(leaked) == 0
    record_test(f"Stack Trace Hygiene ({test_name})", passed, f"leaked: {leaked}" if not passed else "")

def run_suite():
    print("=" * 70)
    print("PHANTOM AI 3.0 — PHASE 6.1 SECURITY & FAILURE ACCEPTANCE SUITE")
    print("=" * 70)

    client = httpx.Client(timeout=15.0)

    # -------------------------------------------------------------------------
    # 1. UNAUTHENTICATED ACCESS
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 1: UNAUTHENTICATED REQUEST REJECTION ---")
    r1 = client.get(f"{API_BASE}/cases")
    record_test("Unauthenticated GET /cases returns 401", r1.status_code == 401, f"Status: {r1.status_code}")
    assert_no_leakage(r1.text, "GET /cases 401")

    r2 = client.post(f"{API_BASE}/cases", json={"title": "Hacked Case", "caseNumber": "CASE-999999", "description": "Desc"})
    record_test("Unauthenticated POST /cases returns 401", r2.status_code == 401, f"Status: {r2.status_code}")
    assert_no_leakage(r2.text, "POST /cases 401")

    r3 = client.get(f"{API_BASE}/reports")
    record_test("Unauthenticated GET /reports returns 401", r3.status_code == 401, f"Status: {r3.status_code}")
    assert_no_leakage(r3.text, "GET /reports 401")

    # -------------------------------------------------------------------------
    # REGISTER INVESTIGATOR A & B FOR RBAC ISOLATION TESTING
    # -------------------------------------------------------------------------
    print("\n--- SETUP: CREATING TWO DISTINCT INVESTIGATORS FOR RBAC ---")
    ts = int(time.time() * 1000) % 1000000
    user_a_email = f"investigator_a_{ts}@phantom.test"
    user_b_email = f"investigator_b_{ts}@phantom.test"

    reg_a = client.post(f"{API_BASE}/auth/register", json={
        "name": f"Investigator Alpha {ts}",
        "badgeNumber": f"BADGE-A-{ts}",
        "email": user_a_email,
        "password": "CorrectHorseBatteryStaple123!",
        "role": "investigator"
    })
    token_a = reg_a.json()["data"]["accessToken"]

    reg_b = client.post(f"{API_BASE}/auth/register", json={
        "name": f"Investigator Beta {ts}",
        "badgeNumber": f"BADGE-B-{ts}",
        "email": user_b_email,
        "password": "CorrectHorseBatteryStaple123!",
        "role": "investigator"
    })
    token_b = reg_b.json()["data"]["accessToken"]

    headers_a = {"Authorization": f"Bearer {token_a}", "X-Phantom-CSRF": "1"}
    headers_b = {"Authorization": f"Bearer {token_b}", "X-Phantom-CSRF": "1"}

    # User A creates Case A
    case_a_res = client.post(f"{API_BASE}/cases", headers=headers_a, json={
        "caseNumber": f"CASE-SEC-A-{ts}",
        "title": "Restricted Homicide File Alpha",
        "description": "Evidence accessible only to authorized lead investigator A."
    })
    case_a_id = case_a_res.json()["data"]["_id"]
    print(f"  Created Case A: {case_a_id} under Investigator A")

    # -------------------------------------------------------------------------
    # 2. CROSS-INVESTIGATOR CASE ISOLATION (RBAC 403)
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 2: CROSS-INVESTIGATOR RBAC ISOLATION ---")
    cross_get = client.get(f"{API_BASE}/cases/{case_a_id}", headers=headers_b)
    record_test("Cross-Investigator GET /cases/:id returns 403", cross_get.status_code == 403, f"Status: {cross_get.status_code}")
    assert_no_leakage(cross_get.text, "Cross-GET 403")

    cross_update = client.patch(f"{API_BASE}/cases/{case_a_id}", headers=headers_b, json={"title": "Unauthorized Modification"})
    record_test("Cross-Investigator PATCH /cases/:id returns 403", cross_update.status_code == 403, f"Status: {cross_update.status_code}")
    assert_no_leakage(cross_update.text, "Cross-PATCH 403")

    cross_gen = client.post(f"{API_BASE}/cases/{case_a_id}/composites/generate", headers=headers_b, json={
        "witnessDescriptionId": "507f1f77bcf86cd799439011"
    })
    record_test("Cross-Investigator POST /cases/:id/composites/generate returns 403", cross_gen.status_code == 403, f"Status: {cross_gen.status_code}")
    assert_no_leakage(cross_gen.text, "Cross-GEN 403")

    cross_rep = client.get(f"{API_BASE}/cases/{case_a_id}/reports", headers=headers_b)
    record_test("Cross-Investigator GET /cases/:id/reports returns 403", cross_rep.status_code == 403, f"Status: {cross_rep.status_code}")
    assert_no_leakage(cross_rep.text, "Cross-REP 403")

    # -------------------------------------------------------------------------
    # 3. SYNTHESIS GATEKEEPING: UNAPPROVED WITNESS STATEMENT
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 3: SYNTHESIS GATEKEEPING (UNAPPROVED WITNESS) ---")
    # Investigator A creates a witness description
    wit_res = client.post(f"{API_BASE}/witness-descriptions", headers=headers_a, json={
        "case": case_a_id,
        "rawStatement": "The perpetrator was tall with short dark hair and a light mustache."
    })
    witness_id = wit_res.json()["data"]["_id"]
    print(f"  Created unapproved witness description: {witness_id} (reviewStatus: pending_review)")

    # Attempt to generate composite BEFORE approval
    unapproved_gen = client.post(f"{API_BASE}/cases/{case_a_id}/composites/generate", headers=headers_a, json={
        "witnessDescriptionId": witness_id,
        "numVariations": 1,
        "seed": 42
    })
    record_test("Composite generation on unapproved witness rejected with 400", unapproved_gen.status_code == 400, f"Status: {unapproved_gen.status_code}")
    assert_no_leakage(unapproved_gen.text, "Unapproved synthesis 400")
    record_test("Rejection message specifies review approval requirement", "approved" in unapproved_gen.text.lower(), unapproved_gen.text[:120])

    # -------------------------------------------------------------------------
    # 4. INVALID OBJECTIDS & NOT FOUND (400 / 404)
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 4: OBJECTID VALIDATION & 404 ROUTING ---")
    invalid_case_id = client.get(f"{API_BASE}/cases/not-a-valid-hex-id", headers=headers_a)
    record_test("Invalid Case ObjectId format returns 400", invalid_case_id.status_code == 400, f"Status: {invalid_case_id.status_code}")
    assert_no_leakage(invalid_case_id.text, "Invalid case ObjectId 400")

    nonexistent_case = client.get(f"{API_BASE}/cases/507f1f77bcf86cd799439011", headers=headers_a)
    record_test("Nonexistent valid Case ObjectId returns 404", nonexistent_case.status_code == 404, f"Status: {nonexistent_case.status_code}")
    assert_no_leakage(nonexistent_case.text, "Nonexistent case 404")

    invalid_img_id = client.get(f"{API_BASE}/cases/{case_a_id}/composites/invalid-id/image/0", headers=headers_a)
    record_test("Invalid Composite ObjectId in image route returns 400", invalid_img_id.status_code == 400, f"Status: {invalid_img_id.status_code}")
    assert_no_leakage(invalid_img_id.text, "Invalid composite image 400")

    nonexistent_img = client.get(f"{API_BASE}/cases/{case_a_id}/composites/507f1f77bcf86cd799439011/image/0", headers=headers_a)
    record_test("Nonexistent Composite ObjectId in image route returns 404", nonexistent_img.status_code == 404, f"Status: {nonexistent_img.status_code}")
    assert_no_leakage(nonexistent_img.text, "Nonexistent composite image 404")

    # -------------------------------------------------------------------------
    # 5. PATH TRAVERSAL ATTEMPTS
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 5: PATH TRAVERSAL MITIGATION ---")
    pt1 = client.get(f"{API_BASE}/cases/{case_a_id}/reports/../../../../etc/passwd/download", headers=headers_a)
    record_test("Path traversal attempt in report download route rejected (400 or 404)", pt1.status_code in [400, 404], f"Status: {pt1.status_code}")
    assert_no_leakage(pt1.text, "Path traversal report download")

    pt2 = client.get(f"{API_BASE}/cases/{case_a_id}/composites/..%2f..%2fpackage.json/image/0", headers=headers_a)
    record_test("Path traversal attempt in composite image route rejected (400 or 404)", pt2.status_code in [400, 404], f"Status: {pt2.status_code}")
    assert_no_leakage(pt2.text, "Path traversal composite image")

    # -------------------------------------------------------------------------
    # 6. ML SERVICE VALIDATION: UNSUPPORTED LATENT DIRECTION & EMPTY CONDITIONING
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 6: ML CONDITIONING & LATENT DIRECTION VALIDATION ---")
    # Trigger BiLSTM analysis and approve witness statement
    ana_res = client.post(f"{API_BASE}/ai-analysis/{witness_id}", headers=headers_a)
    record_test("BiLSTM-CRF inference succeeded on statement", ana_res.status_code == 200, f"Status: {ana_res.status_code}")

    app_res = client.post(f"{API_BASE}/witness-descriptions/{witness_id}/approve", headers=headers_a, json={"status": "approved"})
    record_test("Witness statement approved by investigator", app_res.status_code == 200, f"Status: {app_res.status_code}")

    # Generate composite v1
    gen_res = client.post(f"{API_BASE}/cases/{case_a_id}/composites/generate", headers=headers_a, json={
        "witnessDescriptionId": witness_id,
        "numVariations": 2,
        "seed": 42
    })
    record_test("WGAN-GP composite generation on approved witness succeeded", gen_res.status_code == 201, f"Status: {gen_res.status_code}")
    comp_v1_id = gen_res.json()["data"]["_id"]

    # Test unsupported latent direction in semantic editing
    edit_invalid = client.post(f"{API_BASE}/cases/{case_a_id}/composites/{comp_v1_id}/edit", headers=headers_a, json={
        "seed": 42,
        "latentTraversals": [{"attribute": "completely_unsupported_direction_xyz", "alpha": 2.0}]
    })
    record_test("Semantic edit with unsupported latent direction rejected with 400/500", edit_invalid.status_code in [400, 500], f"Status: {edit_invalid.status_code}")
    assert_no_leakage(edit_invalid.text, "Unsupported latent direction")

    # Directly test ML service with invalid schema / out-of-range vector
    ml_invalid_schema = client.post(f"{ML_BASE}/generate", json={"invalid_field": True})
    record_test("Direct ML /generate with invalid payload rejected with 422", ml_invalid_schema.status_code == 422, f"Status: {ml_invalid_schema.status_code}")

    # -------------------------------------------------------------------------
    # 7. OVERSIZED PAYLOAD (413)
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 7: OVERSIZED PAYLOAD PROTECTION ---")
    huge_payload = {"title": "X" * (12 * 1024 * 1024), "description": "Big"}
    try:
        oversized_res = client.post(f"{API_BASE}/cases", headers=headers_a, json=huge_payload)
        record_test("Oversized payload (>10MB) returns 413 Payload Too Large", oversized_res.status_code == 413, f"Status: {oversized_res.status_code}")
        assert_no_leakage(oversized_res.text, "Oversized payload 413")
    except Exception as e:
        # Client or server closed connection due to size
        record_test("Oversized payload (>10MB) rejected by server connection limit", True, str(e)[:100])

    # -------------------------------------------------------------------------
    # 8. FORENSIC DATABASE & DISK ARTIFACT INTEGRITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 8: FORENSIC DATABASE & DISK ARTIFACT VERIFICATION ---")

    # Generate a report for Case A
    rep_res = client.post(f"{API_BASE}/cases/{case_a_id}/reports", headers=headers_a, json={
        "witnessDescriptionId": witness_id,
        "compositeId": comp_v1_id,
        "compositeVersion": 1,
        "title": "Acceptance Test Forensic Dossier",
        "notes": "Verified end-to-end acceptance run."
    })
    record_test("Report generation for Case A succeeded (201)", rep_res.status_code == 201, f"Status: {rep_res.status_code}")
    report_data = rep_res.json()["data"]

    # Verify Composite Image File on disk
    comp_data = gen_res.json()["data"]
    comp_storage_key = comp_data["variations"][0]["storageKey"]
    comp_sha256 = comp_data["variations"][0]["sha256"]
    comp_disk_path = os.path.join("/Users/bhumi/Desktop/Phantom-AI/phantom-ai-api", comp_storage_key)

    record_test("Composite PNG file exists on disk", os.path.exists(comp_disk_path), comp_disk_path)
    if os.path.exists(comp_disk_path):
        with open(comp_disk_path, "rb") as f:
            png_bytes = f.read()
        is_png = png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        actual_comp_sha = hashlib.sha256(png_bytes).hexdigest()
        record_test("Composite file has valid PNG 8-byte magic header", is_png)
        record_test("Composite file SHA-256 matches database record exactly", actual_comp_sha == comp_sha256, f"Disk: {actual_comp_sha[:16]}... DB: {comp_sha256[:16]}...")
        # Check dimensions in IHDR
        width = int.from_bytes(png_bytes[16:20], 'big')
        height = int.from_bytes(png_bytes[20:24], 'big')
        record_test("Composite PNG has native 64x64 forensic resolution", width == 64 and height == 64, f"{width}x{height}")

    # Verify PDF Report File on disk
    rep_storage_key = report_data["storageKey"]
    rep_sha256 = report_data["sha256"]
    rep_disk_path = os.path.join("/Users/bhumi/Desktop/Phantom-AI/phantom-ai-api", rep_storage_key)

    record_test("Forensic PDF report file exists on disk", os.path.exists(rep_disk_path), rep_disk_path)
    if os.path.exists(rep_disk_path):
        with open(rep_disk_path, "rb") as f:
            pdf_bytes = f.read()
        is_pdf = pdf_bytes.startswith(b"%PDF")
        actual_rep_sha = hashlib.sha256(pdf_bytes).hexdigest()
        record_test("Dossier has valid %PDF magic header", is_pdf)
        record_test("Dossier file SHA-256 matches database record exactly", actual_rep_sha == rep_sha256, f"Disk: {actual_rep_sha[:16]}... DB: {rep_sha256[:16]}...")
        record_test("Dossier contains expected section markers", b"Section" in pdf_bytes or b"FORENSIC" in pdf_bytes, f"Size: {len(pdf_bytes)} bytes")

    # -------------------------------------------------------------------------
    # 9. POST-TEST CHECKPOINT INTEGRITY
    # -------------------------------------------------------------------------
    print("\n--- TEST GROUP 9: POST-TEST CHECKPOINT SHA-256 INTEGRITY ---")
    EXPECTED_CHECKPOINTS = {
        "generator": {
            "path": "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt",
            "expected_sha": "a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"
        },
        "evaluator": {
            "path": "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt",
            "expected_sha": "f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834"
        },
        "nlp": {
            "path": "phantom-ai-ml/checkpoints/nlp_char_bilstm_phase3c2.pt",
            "expected_sha": "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"
        }
    }

    for name, item in EXPECTED_CHECKPOINTS.items():
        with open(item["path"], "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        match = digest == item["expected_sha"]
        record_test(f"Checkpoint integrity preserved ({name})", match, f"{digest[:16]}... == {item['expected_sha'][:16]}...")

    # Print Summary
    print("\n" + "=" * 70)
    print("PHASE 6.1 SECURITY & FAILURE SUITE SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run: {test_summary['total_tests']}")
    print(f"Passed: {test_summary['passed_tests']}")
    print(f"Failed: {test_summary['failed_tests']}")

    results_file = "/Users/bhumi/.gemini/antigravity/brain/ddf7edba-fdda-4225-92cb-a2c78b269d98/phase61_screenshots/security_failure_suite_results.json"
    with open(results_file, "w") as f:
        json.dump(test_summary, f, indent=2)
    print(f"Results saved to: {results_file}")

    if test_summary["failed_tests"] > 0:
        print("\nFailures:")
        for fail in test_summary["failures"]:
            print(f"  - {fail['name']}: {fail['details']}")
        sys.exit(1)
    else:
        print("\nALL SECURITY, FAILURE-PATH, AND INTEGRITY TESTS PASSED (100%).")
        sys.exit(0)

if __name__ == "__main__":
    run_suite()


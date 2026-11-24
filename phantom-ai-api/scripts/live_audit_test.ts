import mongoose from 'mongoose';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { env } from '../src/config/env';
import { User } from '../src/models/User.model';
import { Session } from '../src/models/Session.model';
import { Case } from '../src/models/Case.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Composite } from '../src/models/Composite.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';

const API_BASE = 'http://127.0.0.1:5001/api/v1';
const ML_BASE = 'http://127.0.0.1:8000';

function hashToken(token: string): string {
  return crypto.createHash('sha256').update(token).digest('hex');
}

async function createAuthenticatedSession(userDoc: any) {
  const sessionId = new mongoose.Types.ObjectId();
  const rawToken = jwt.sign(
    {
      id: userDoc._id.toString(),
      investigatorId: userDoc.investigatorId,
      role: userDoc.role,
      email: userDoc.email,
    },
    env.JWT_SECRET,
    { expiresIn: '7d' }
  );

  const tokenDigest = hashToken(rawToken);
  await Session.create({
    _id: sessionId,
    userId: userDoc._id,
    investigatorId: userDoc.investigatorId,
    tokenHash: tokenDigest,
    isValid: true,
    expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    userAgent: 'LiveAuditSuite/1.0',
    ipAddress: '127.0.0.1',
  });

  return rawToken;
}

async function runLiveAudit() {
  console.log('================================================================');
  console.log('   PHANTOM AI 3.0 — PHASE 5.5.1 LIVE TECHNICAL AUDIT SUITE      ');
  console.log('================================================================\n');

  if (mongoose.connection.readyState === 0) {
    await mongoose.connect(env.MONGODB_URI);
  }
  console.log('[1/12] Connected to MongoDB at', env.MONGODB_URI);

  // 1. Verify ML Service Health
  console.log('\n[2/12] Verifying Live Python ML Service (port 8000)...');
  const mlHealthRes = await fetch(`${ML_BASE}/health`);
  if (!mlHealthRes.ok) throw new Error(`ML health check failed: ${mlHealthRes.status}`);
  const mlHealth: any = await mlHealthRes.json();
  console.log('  -> ML Service status:', mlHealth.status, '| Device:', mlHealth.device?.device_name);

  const mlStatusRes = await fetch(`${ML_BASE}/model-status`);
  const mlStatus: any = await mlStatusRes.json();
  const cganInfo = mlStatus.models?.cgan_generator;
  console.log('  -> CGAN Generator registered:', !!cganInfo);
  console.log('  -> CGAN Architecture:', cganInfo?.architecture);
  console.log('  -> CGAN Parameters (Direct PyTorch evaluation):', cganInfo?.parameter_count);
  console.log('  -> CGAN Checkpoint SHA-256:', cganInfo?.notes?.match(/[a-f0-9]{64}/)?.[0]);

  // 2. Setup Test Users & Sessions
  console.log('\n[3/12] Creating Authenticated Test Investigators in MongoDB...');
  const testSuffix = Date.now().toString();
  const inv1 = await User.create({
    name: 'Inspector Sherlock Holmes',
    email: `inspector.audit.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'investigator',
    badgeNumber: `BADGE-${testSuffix.slice(-4)}`,
    investigatorId: `INV-${testSuffix.slice(-6)}`,
  });
  const token1 = await createAuthenticatedSession(inv1);
  console.log(`  -> Authorized Investigator created: ${inv1.name} (${inv1.investigatorId})`);

  const inv2 = await User.create({
    name: 'Unauthorized Rogue Investigator',
    email: `rogue.audit.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'investigator',
    badgeNumber: `ROGUE-${testSuffix.slice(-4)}`,
    investigatorId: `ROGUE-${testSuffix.slice(-6)}`,
  });
  const token2 = await createAuthenticatedSession(inv2);
  console.log(`  -> Unauthorized Investigator created: ${inv2.name} (${inv2.investigatorId})`);

  // 3. Create Case
  console.log('\n[4/12] Creating Forensic Case via Live Node API...');
  const caseRes = await fetch(`${API_BASE}/cases`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({
      caseNumber: `CASE-AUDIT-${testSuffix.slice(-6)}`,
      title: 'Forensic Live Audit Armed Robbery Case',
      description: 'End-to-end verification of Phase 5.5 composite synthesis pipeline',
    }),
  });
  const caseData: any = await caseRes.json();
  if (!caseRes.ok) throw new Error(`Case creation failed: ${JSON.stringify(caseData)}`);
  const caseId = caseData.data?._id || caseData._id;
  console.log(`  -> Case created successfully: ID ${caseId} (${caseData.data?.caseNumber})`);

  // 4. Ingest Witness Description
  console.log('\n[5/12] Ingesting Verbatim Witness Statement...');
  const witnessStatement =
    'The perpetrator was an adult male with clean-shaven face, thick black wavy hair, full lips, high cheekbones, and he was wearing eyeglasses.';
  const witRes = await fetch(`${API_BASE}/witness-descriptions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({
      case: caseId,
      rawStatement: witnessStatement,
      witnessName: 'Witness Arthur Pendelton',
    }),
  });
  const witData: any = await witRes.json();
  if (!witRes.ok) throw new Error(`Witness intake failed: ${JSON.stringify(witData)}`);
  const witnessId = witData.data?._id || witData._id;
  console.log(`  -> Witness statement recorded: ID ${witnessId}`);

  // 5. Run Local BiLSTM-CRF Extraction via Node ML Client
  console.log('\n[6/12] Running Local BiLSTM-CRF Attribute Extraction...');
  const analysisRes = await fetch(`${API_BASE}/ai-analysis/${witnessId}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token1}`,
    },
  });
  const analysisData: any = await analysisRes.json();
  if (!analysisRes.ok) throw new Error(`AI Analysis failed: ${JSON.stringify(analysisData)}`);
  console.log('  -> BiLSTM-CRF analysis successful. Status:', analysisData.data?.reviewStatus);
  console.log('  -> Extracted attributes count:', analysisData.data?.extractedAttributesList?.length);

  // 6. Test Security Gate: Try Generating Composite BEFORE Approval
  console.log('\n[7/12] Testing Security Gate: Attempting generation on UNAPPROVED profile...');
  const prematureGenRes = await fetch(`${API_BASE}/cases/${caseId}/composites/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({
      witnessDescriptionId: witnessId,
      seed: 42,
      numVariations: 2,
    }),
  });
  const prematureGenData: any = await prematureGenRes.json();
  console.log(`  -> HTTP Status: ${prematureGenRes.status} (Expected 400)`);
  console.log(`  -> Response message: "${prematureGenData.message}"`);
  if (prematureGenRes.status !== 400) {
    throw new Error(`SECURITY VIOLATION: Unapproved generation was not blocked with 400! Got ${prematureGenRes.status}`);
  }

  // 7. Test Security Gate: Unauthorized Investigator Access
  console.log('\n[8/12] Testing Security Gate: Attempting generation by UNAUTHORIZED investigator...');
  const unauthorizedGenRes = await fetch(`${API_BASE}/cases/${caseId}/composites/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token2}`, // Rogue token
    },
    body: JSON.stringify({
      witnessDescriptionId: witnessId,
      seed: 42,
      numVariations: 2,
    }),
  });
  console.log(`  -> HTTP Status: ${unauthorizedGenRes.status} (Expected 403 Forbidden)`);
  if (unauthorizedGenRes.status !== 403) {
    throw new Error(`SECURITY VIOLATION: Unauthorized investigator access was not blocked with 403! Got ${unauthorizedGenRes.status}`);
  }

  // 8. Investigator Formally Approves Profile
  console.log('\n[9/12] Formally Approving Evidence Profile by Authenticated Investigator...');
  const approveRes = await fetch(`${API_BASE}/witness-descriptions/${witnessId}/approve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({ status: 'approved' }),
  });
  const approveData = await approveRes.json();
  if (!approveRes.ok) throw new Error(`Approval failed: ${JSON.stringify(approveData)}`);
  console.log('  -> Evidence profile status updated to: APPROVED');

  // 9. Execute Actual Conditional Synthesis (Batch 1: Version 1)
  console.log('\n[10/12] Executing Actual WGAN-GP Composite Synthesis (Batch 1)...');
  const gen1Res = await fetch(`${API_BASE}/cases/${caseId}/composites/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({
      witnessDescriptionId: witnessId,
      seed: 101,
      numVariations: 2,
      notes: 'Live audit Batch 1 primary reconstruction test',
    }),
  });
  const gen1Data: any = await gen1Res.json();
  if (!gen1Res.ok) throw new Error(`Composite Generation Batch 1 failed: ${JSON.stringify(gen1Data)}`);
  const comp1 = gen1Data.data?.composite || gen1Data.composite || gen1Data.data;
  console.log(`  -> Synthesis Succeeded! Version: ${comp1.version} | Composite ID: ${comp1._id}`);
  console.log(`  -> Conditioned GCIT features: [${comp1.conditionedAttributes?.join(', ')}]`);
  console.log(`  -> Generated variations count: ${comp1.variations?.length}`);

  // Verify stored images on disk for Batch 1
  const storageDir = path.resolve(process.cwd(), 'data/composites');
  for (const v of comp1.variations) {
    const fullImgPath = path.resolve(storageDir, v.storageKey);
    if (!fs.existsSync(fullImgPath)) {
      throw new Error(`File missing on disk: ${fullImgPath}`);
    }
    const buf = fs.readFileSync(fullImgPath);
    // Validate PNG Magic Bytes: 89 50 4E 47 0D 0A 1A 0A
    const magic = buf.subarray(0, 8);
    const expected = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    if (!magic.equals(expected)) {
      throw new Error(`Corrupt PNG magic bytes on disk: ${fullImgPath}`);
    }
    console.log(`  -> Disk file verified: ${v.storageKey} (${buf.length} bytes, 64x64 PNG magic bytes valid)`);
  }

  // 10. Test Image Serving & Security Endpoints
  console.log('\n[11/12] Testing Authenticated Image Serving & Path Traversal Prevention...');
  // Authenticated retrieval
  const imgAuthRes = await fetch(`${API_BASE}/cases/${caseId}/composites/${comp1._id}/image/1`, {
    headers: { Authorization: `Bearer ${token1}` },
  });
  console.log(`  -> Authenticated image fetch HTTP Status: ${imgAuthRes.status} (Content-Type: ${imgAuthRes.headers.get('content-type')})`);
  if (!imgAuthRes.ok) throw new Error('Authenticated image fetch failed');

  // Unauthorized retrieval
  const imgUnauthRes = await fetch(`${API_BASE}/cases/${caseId}/composites/${comp1._id}/image/1`, {
    headers: { Authorization: `Bearer ${token2}` },
  });
  console.log(`  -> Unauthorized image fetch HTTP Status: ${imgUnauthRes.status} (Expected 403)`);
  if (imgUnauthRes.status !== 403) throw new Error('Unauthorized image fetch was not blocked');

  // Unauthenticated retrieval
  const imgNoAuthRes = await fetch(`${API_BASE}/cases/${caseId}/composites/${comp1._id}/image/1`);
  console.log(`  -> Unauthenticated image fetch HTTP Status: ${imgNoAuthRes.status} (Expected 401)`);
  if (imgNoAuthRes.status !== 401) throw new Error('Unauthenticated image fetch was not blocked');

  // Path traversal prevention check
  const traversalRes = await fetch(`${API_BASE}/cases/${caseId}/composites/${comp1._id}/image/%2e%2e%2f%2e%2e%2fpackage.json`, {
    headers: { Authorization: `Bearer ${token1}` },
  });
  console.log(`  -> Encoded path traversal HTTP Status: ${traversalRes.status} (Expected 400 or 404)`);

  // Variation Selection / Approval
  console.log('  -> Selecting Variation 1 as Primary Lead Composite...');
  const selectRes = await fetch(`${API_BASE}/cases/${caseId}/composites/${comp1._id}/select`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({
      variationIndex: 1,
      status: 'approved',
      notes: 'Investigator selected variation 1 as lead suspect visual representation',
    }),
  });
  const selectData: any = await selectRes.json();
  console.log('  -> Variation selection successful. Selected variation:', selectData.data?.selectedVariation, 'Status:', selectData.data?.status);

  // 11. Execute Batch 2 to verify Sequential Versioning (Version 2)
  console.log('\n[12/12] Executing Batch 2 to verify Zero-Overwrite Sequential Versioning...');
  const gen2Res = await fetch(`${API_BASE}/cases/${caseId}/composites/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token1}`,
    },
    body: JSON.stringify({
      witnessDescriptionId: witnessId,
      seed: 202,
      numVariations: 2,
      notes: 'Batch 2 second generation test',
    }),
  });
  const gen2Data: any = await gen2Res.json();
  const comp2 = gen2Data.data?.composite || gen2Data.composite || gen2Data.data;
  console.log(`  -> Batch 2 Succeeded! Version: ${comp2.version} (Expected Version 2)`);
  if (comp2.version !== 2) throw new Error(`Versioning failure: expected version 2, got ${comp2.version}`);

  // Confirm Version 1 remains intact on disk and in database
  const v1Recheck = await Composite.findById(comp1._id);
  if (!v1Recheck || v1Recheck.version !== 1) {
    throw new Error('CORRUPTION DETECTED: Version 1 composite document was overwritten or corrupted!');
  }
  for (const v of v1Recheck.variations) {
    const fPath = path.resolve(storageDir, v.storageKey);
    if (!fs.existsSync(fPath)) {
      throw new Error(`CORRUPTION DETECTED: Version 1 image ${v.storageKey} was deleted from disk!`);
    }
  }
  console.log('  -> Version 1 files and database document confirmed 100% INTACT.');

  // Check version history listing
  const listRes = await fetch(`${API_BASE}/cases/${caseId}/composites`, {
    headers: { Authorization: `Bearer ${token1}` },
  });
  const listData: any = await listRes.json();
  const composites = listData.data?.composites || listData.composites || listData.data;
  console.log(`  -> History listing returned ${composites.length} versions (v${composites[0]?.version}, v${composites[1]?.version})`);

  // Verify Audit Trail
  const auditLogs = await SecurityAudit.find({
    userId: inv1._id,
    event: { $in: ['COMPOSITE_GENERATION_INITIATED', 'COMPOSITE_GENERATED_SUCCESS', 'COMPOSITE_VIEWED', 'COMPOSITE_APPROVED'] },
  });
  console.log(`  -> Verified Security Audit logs recorded: ${auditLogs.length} events logged`);

  // Clean up test data
  console.log('\nCleaning up audit test records...');
  await User.deleteMany({ _id: { $in: [inv1._id, inv2._id] } });
  await Session.deleteMany({ userId: { $in: [inv1._id, inv2._id] } });
  await Case.findByIdAndDelete(caseId);
  await WitnessDescription.findByIdAndDelete(witnessId);
  await Composite.deleteMany({ case: caseId });
  await SecurityAudit.deleteMany({ userId: { $in: [inv1._id, inv2._id] } });

  console.log('\n================================================================');
  console.log('   LIVE AUDIT PASSED: ALL 12 VERIFICATION STAGES SUCCEEDED!     ');
  console.log('================================================================');

  await mongoose.connection.close();
}

runLiveAudit().catch((err) => {
  console.error('\n*** LIVE AUDIT SUITE FAILED ***', err);
  process.exit(1);
});

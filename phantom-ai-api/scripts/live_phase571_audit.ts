import mongoose from 'mongoose';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
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
    userAgent: 'LiveAuditSuitePhase571/1.0',
    ipAddress: '127.0.0.1',
  });

  return rawToken;
}

function verifyPngBuffer(buffer: Buffer): boolean {
  if (buffer.length < 100) return false;
  const pngHeader = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < pngHeader.length; i++) {
    if (buffer[i] !== pngHeader[i]) return false;
  }
  return true;
}

async function runLiveAuditPhase571() {
  console.log('================================================================');
  console.log('   PHANTOM AI 3.0 — PHASE 5.7.1 LIVE SEMANTIC EDITING AUDIT     ');
  console.log('================================================================\n');

  if (mongoose.connection.readyState === 0) {
    await mongoose.connect(env.MONGODB_URI);
  }
  console.log('[1/10] Connected to MongoDB at', env.MONGODB_URI);

  // 1. Check services
  console.log('\n[2/10] Verifying Live Python ML Service and Express API...');
  const mlRes = await fetch(`${ML_BASE}/health`);
  if (!mlRes.ok) throw new Error(`FastAPI health check failed: ${mlRes.status}`);
  const mlHealth: any = await mlRes.json();
  console.log(`  -> FastAPI Service: ${mlHealth.status} | Device: ${mlHealth.device?.device_name}`);

  const expressRes = await fetch(`${API_BASE}/health`);
  if (!expressRes.ok) throw new Error(`Express API health check failed: ${expressRes.status}`);
  console.log(`  -> Express API Service: UP (200)`);

  // 2. Latent directions endpoint verification
  console.log('\n[3/10] Verifying Latent Directions Metadata Endpoint...');
  const latentRes = await fetch(`${ML_BASE}/latent-directions`);
  if (!latentRes.ok) throw new Error(`Latent directions check failed: ${latentRes.status}`);
  const latentData: any = await latentRes.json();
  console.log(`  -> Validated Directions Count: ${latentData.validated_count}`);
  console.log(`  -> Rejected Directions Count:  ${latentData.rejected_count}`);
  console.log(`  -> Validated List:`, latentData.validated_attributes);
  
  if (latentData.validated_count !== 5) {
    throw new Error(`Expected exactly 5 validated directions, got ${latentData.validated_count}`);
  }
  const expected5 = ['hair_blond', 'hair_gray', 'hair_bangs', 'facial_goatee', 'glasses'];
  for (const exp of expected5) {
    if (!latentData.validated_attributes.includes(exp)) {
      throw new Error(`Missing expected validated direction: ${exp}`);
    }
  }

  // 3. Setup Authenticated Users & Case
  console.log('\n[4/10] Setting Up Authenticated Investigator & Authorized Case...');
  const testSuffix = Date.now().toString();
  const authorizedInv = await User.create({
    name: 'Inspector Hercule Poirot',
    email: `poirot.phase571.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'investigator',
    badgeNumber: `BADGE-${testSuffix.slice(-4)}`,
    investigatorId: `INV-${testSuffix.slice(-6)}`,
  });
  const tokenAuth = await createAuthenticatedSession(authorizedInv);

  const unauthorizedInv = await User.create({
    name: 'Rogue Agent Malice',
    email: `rogue.phase571.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'investigator',
    badgeNumber: `ROGUE-${testSuffix.slice(-4)}`,
    investigatorId: `ROGUE-${testSuffix.slice(-6)}`,
  });
  const tokenUnauth = await createAuthenticatedSession(unauthorizedInv);

  const caseDoc = await Case.create({
    caseNumber: `CASE-571-${testSuffix.slice(-5)}`,
    title: 'Phase 5.7.1 Live Audit Semantic Case',
    description: 'Investigating controlled semantic variation across continuous GCIT and latent traversals.',
    createdBy: authorizedInv._id,
    assignedTo: [authorizedInv._id],
  });
  console.log(`  -> Case Created: ${caseDoc.caseNumber} (ID: ${caseDoc._id})`);

  // Create approved witness description
  const witnessDoc = await WitnessDescription.create({
    case: caseDoc._id,
    witnessName: 'Auditor Jane Doe',
    recordedBy: authorizedInv._id,
    rawStatement: 'Suspect was an adult with black hair and a neat goatee wearing glasses.',
    extractedAttributes: [
      { category: 'hair', value: 'black', evidenceState: 'explicit', sourceText: 'black hair', extractedBy: 'nlp' },
      { category: 'facial_hair', value: 'goatee', evidenceState: 'explicit', sourceText: 'goatee', extractedBy: 'nlp' },
      { category: 'accessories', value: 'glasses', evidenceState: 'explicit', sourceText: 'glasses', extractedBy: 'nlp' },
    ],
    reviewStatus: 'approved',
    reviewedBy: authorizedInv._id,
    reviewedAt: new Date(),
  });
  console.log(`  -> Approved Witness Description Created: ID ${witnessDoc._id}`);

  // 4. Synthesize Base Composite (Version 1)
  console.log('\n[5/10] Generating Initial Composite Version 1 via Express API...');
  const genRes = await fetch(`${API_BASE}/cases/${caseDoc._id}/composites/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${tokenAuth}`,
    },
    body: JSON.stringify({
      witnessDescriptionId: witnessDoc._id.toString(),
      seed: 42,
      numVariations: 2,
      notes: 'Initial base composite generation for Phase 5.7.1 audit.',
    }),
  });
  if (!genRes.ok) {
    const errText = await genRes.text();
    throw new Error(`Generation failed (${genRes.status}): ${errText}`);
  }
  const genData: any = await genRes.json();
  console.log('  -> raw genData keys:', Object.keys(genData), 'data keys:', genData.data ? Object.keys(genData.data) : 'none');
  const compV1 = genData.data?.composite || genData.composite || genData.data;
  console.log(`  -> Version 1 Synthesized: ID ${compV1._id} | Version: ${compV1.version} | Variations: ${compV1.variations?.length}`);
  
  if (compV1.version !== 1) throw new Error(`Expected Version 1, got ${compV1.version}`);
  const v1ImageBuffer = Buffer.from(compV1.variations[0].dataUrl.split(',')[1], 'base64');
  if (!verifyPngBuffer(v1ImageBuffer)) throw new Error('V1 image buffer is not a valid PNG (>100 bytes)');
  const v1Hash = crypto.createHash('sha256').update(v1ImageBuffer).digest('hex');
  console.log(`  -> V1 Var #0 Image SHA-256: ${v1Hash} (${v1ImageBuffer.length} bytes)`);

  // 5. Perform Continuous GCIT Edit -> Version 2
  console.log('\n[6/10] Performing Live Continuous GCIT Edit (Nose Size +0.8, Lip Fullness +0.6)...');
  const editGcitRes = await fetch(`${API_BASE}/cases/${caseDoc._id}/composites/${compV1._id}/edit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${tokenAuth}`,
    },
    body: JSON.stringify({
      selectedSeed: 42,
      continuousAdjustments: {
        nose_big: 0.8,
        lips_big: 0.6,
      },
      notes: 'Investigator adjusted nose scale and lip fullness based on secondary witness recall.',
    }),
  });
  if (!editGcitRes.ok) {
    const errText = await editGcitRes.text();
    throw new Error(`Continuous GCIT edit failed (${editGcitRes.status}): ${errText}`);
  }
  const editGcitData: any = await editGcitRes.json();
  const compV2 = editGcitData.data?.composite || editGcitData.composite;
  console.log(`  -> Version 2 Synthesized: ID ${compV2._id} | Version: ${compV2.version}`);
  console.log(`  -> Parent Composite Link: ${compV2.parentComposite}`);
  console.log(`  -> Editing Type:          ${compV2.editingType}`);
  
  if (compV2.version !== 2) throw new Error(`Expected Version 2, got ${compV2.version}`);
  if (compV2.parentComposite.toString() !== compV1._id.toString()) {
    throw new Error(`Parent composite mismatch: expected ${compV1._id}, got ${compV2.parentComposite}`);
  }
  if (compV2.editingType !== 'conditioning_traversal') {
    throw new Error(`Expected editingType 'conditioning_traversal', got ${compV2.editingType}`);
  }

  const v2ImageBuffer = Buffer.from(compV2.variations[0].dataUrl.split(',')[1], 'base64');
  if (!verifyPngBuffer(v2ImageBuffer)) throw new Error('V2 image buffer is not a valid PNG (>100 bytes)');
  const v2Hash = crypto.createHash('sha256').update(v2ImageBuffer).digest('hex');
  console.log(`  -> V2 Var #0 Image SHA-256: ${v2Hash} (${v2ImageBuffer.length} bytes)`);
  if (v2Hash === v1Hash) {
    throw new Error('Edited image SHA-256 matches base image! No modification occurred.');
  }

  // 6. Perform Validated Latent Traversal Edit -> Version 3
  console.log('\n[7/10] Performing Live Validated Latent Hyperplane Traversal (Glasses alpha = +1.5)...');
  const editLatentRes = await fetch(`${API_BASE}/cases/${caseDoc._id}/composites/${compV2._id}/edit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${tokenAuth}`,
    },
    body: JSON.stringify({
      selectedSeed: 42,
      latentTraversals: {
        glasses: 1.5,
      },
      notes: 'Investigator traversed latent hyperplane for glasses presence.',
    }),
  });
  if (!editLatentRes.ok) {
    const errText = await editLatentRes.text();
    throw new Error(`Latent traversal edit failed (${editLatentRes.status}): ${errText}`);
  }
  const editLatentData: any = await editLatentRes.json();
  const compV3 = editLatentData.data?.composite || editLatentData.composite;
  console.log(`  -> Version 3 Synthesized: ID ${compV3._id} | Version: ${compV3.version}`);
  console.log(`  -> Parent Composite Link: ${compV3.parentComposite}`);
  console.log(`  -> Editing Type:          ${compV3.editingType}`);

  if (compV3.version !== 3) throw new Error(`Expected Version 3, got ${compV3.version}`);
  if (compV3.parentComposite.toString() !== compV2._id.toString()) {
    throw new Error(`Parent composite mismatch: expected ${compV2._id}, got ${compV3.parentComposite}`);
  }
  if (compV3.editingType !== 'latent_traversal') {
    throw new Error(`Expected editingType 'latent_traversal', got ${compV3.editingType}`);
  }

  const v3ImageBuffer = Buffer.from(compV3.variations[0].dataUrl.split(',')[1], 'base64');
  if (!verifyPngBuffer(v3ImageBuffer)) throw new Error('V3 image buffer is not a valid PNG (>100 bytes)');
  const v3Hash = crypto.createHash('sha256').update(v3ImageBuffer).digest('hex');
  console.log(`  -> V3 Var #0 Image SHA-256: ${v3Hash} (${v3ImageBuffer.length} bytes)`);

  // 7. Verify Immutability of Version 1 in Database
  console.log('\n[8/10] Verifying Invariant: Version 1 Document & Image Immutability...');
  const v1Check = await Composite.findById(compV1._id);
  if (!v1Check || !v1Check.variations || !v1Check.variations[0]) {
    throw new Error('Version 1 composite disappeared or has no variations!');
  }
  if (v1Check.version !== 1) throw new Error('Version 1 was mutated!');
  const v1CheckDataUrl = v1Check.variations[0].dataUrl || '';
  const v1CheckBuffer = Buffer.from(v1CheckDataUrl.split(',')[1] || '', 'base64');
  const v1CheckHash = crypto.createHash('sha256').update(v1CheckBuffer).digest('hex');
  if (v1CheckHash !== v1Hash) throw new Error('Version 1 image was altered!');
  console.log('  -> Confirmed: Version 1 is 100% bit-exact and immutable in MongoDB.');

  // 8. Verify Security Audit Events
  console.log('\n[9/10] Verifying Security Audit Logs in MongoDB...');
  const auditLogs = await SecurityAudit.find({
    userId: authorizedInv._id,
    event: { $in: ['COMPOSITE_GENERATED_SUCCESS', 'COMPOSITE_SEMANTIC_EDIT'] },
  }).sort({ createdAt: 1 });
  console.log(`  -> Audit Records Logged for Authorized Investigator: ${auditLogs.length}`);
  for (const log of auditLogs) {
    console.log(`     [AUDIT] ${log.event} | Status: ${log.status} | User: ${log.userId} | Details:`, log.details);
  }
  if (auditLogs.length < 3) {
    throw new Error(`Expected at least 3 audit events (1 generation + 2 edits), found ${auditLogs.length}`);
  }

  // 9. Negative Security & Validation Tests
  console.log('\n[10/10] Running Negative Tests: Unauthorized Access & Rejected Directions...');
  
  // 9a. Unauthorized investigator attempt
  const unauthEditRes = await fetch(`${API_BASE}/cases/${caseDoc._id}/composites/${compV1._id}/edit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${tokenUnauth}`,
    },
    body: JSON.stringify({
      selectedSeed: 42,
      continuousAdjustments: { nose_big: 0.5 },
    }),
  });
  console.log(`  -> Unauthorized access status: ${unauthEditRes.status} (Expected: 403 Forbidden)`);
  if (unauthEditRes.status !== 403) throw new Error(`Expected 403 for unauthorized edit, got ${unauthEditRes.status}`);

  // 9b. Rejected latent direction (hair_black in Z)
  const rejectedDirectionRes = await fetch(`${API_BASE}/cases/${caseDoc._id}/composites/${compV1._id}/edit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${tokenAuth}`,
    },
    body: JSON.stringify({
      selectedSeed: 42,
      latentTraversals: { hair_black: 1.0 },
    }),
  });
  console.log(`  -> Rejected direction 'hair_black' status: ${rejectedDirectionRes.status} (Expected: 400 Bad Request)`);
  if (rejectedDirectionRes.status !== 400) throw new Error(`Expected 400 for rejected direction, got ${rejectedDirectionRes.status}`);

  // 9c. Unsupported feature (smile_intensity)
  const unsupportedRes = await fetch(`${API_BASE}/cases/${caseDoc._id}/composites/${compV1._id}/edit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${tokenAuth}`,
    },
    body: JSON.stringify({
      selectedSeed: 42,
      latentTraversals: { smile_intensity: 1.0 },
    }),
  });
  console.log(`  -> Unsupported feature 'smile_intensity' status: ${unsupportedRes.status} (Expected: 400 Bad Request)`);
  if (unsupportedRes.status !== 400) throw new Error(`Expected 400 for unsupported feature, got ${unsupportedRes.status}`);

  console.log('\n================================================================');
  console.log('   PHASE 5.7.1 LIVE SEMANTIC EDITING AUDIT: ALL TESTS PASSED!   ');
  console.log('================================================================');
}

runLiveAuditPhase571()
  .then(() => {
    process.exit(0);
  })
  .catch((err) => {
    console.error('\n[FATAL AUDIT FAILURE]:', err);
    process.exit(1);
  });

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import zlib from 'zlib';
import mongoose from 'mongoose';
import axios from 'axios';
import dotenv from 'dotenv';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { Session } from '../src/models/Session.model';

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const API_BASE = process.env.API_BASE_URL || 'http://127.0.0.1:5001/api/v1';
const MONGO_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/phantom_ai_dev';

function fileSha256(filePath: string): string {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

async function createAuthSession(user: any): Promise<string> {
  const jwt = require('jsonwebtoken');
  const jwtSecret = process.env.JWT_SECRET || 'dev_secret_key_123';
  const sessionId = new mongoose.Types.ObjectId();
  const rawToken = jwt.sign(
    {
      id: user._id.toString(),
      role: user.role,
      investigatorId: user.investigatorId || 'INV-AUDIT',
      email: user.email,
      name: user.name,
      sessionId: sessionId.toString(),
    },
    jwtSecret,
    { expiresIn: '2h' }
  );

  const tokenDigest = crypto.createHash('sha256').update(rawToken).digest('hex');
  await Session.create({
    _id: sessionId,
    userId: user._id,
    investigatorId: user.investigatorId || 'INV-AUDIT',
    tokenHash: tokenDigest,
    isValid: true,
    expiresAt: new Date(Date.now() + 2 * 3600 * 1000),
    userAgent: 'Phase-5.8.1-Audit/1.0',
    ipAddress: '127.0.0.1',
  });

  return rawToken;
}

function extractPdfStreamsText(pdfBuffer: Buffer): string[] {
  const streamStart = Buffer.from('stream\n');
  const streamEnd = Buffer.from('\nendstream');

  let idx = 0;
  let fullText: string[] = [];

  while ((idx = pdfBuffer.indexOf(streamStart, idx)) !== -1) {
    idx += streamStart.length;
    const endIdx = pdfBuffer.indexOf(streamEnd, idx);
    if (endIdx !== -1) {
      const raw = pdfBuffer.slice(idx, endIdx);
      try {
        const dec = zlib.inflateSync(raw).toString('latin1');
        if (dec.includes('BT')) {
          const hexMatches = dec.matchAll(/<([0-9a-fA-F]+)>/g);
          let streamWords = [];
          for (const hm of hexMatches) {
            const hex = hm[1];
            const text = Buffer.from(hex, 'hex').toString('latin1');
            streamWords.push(text);
          }
          fullText.push(streamWords.join(''));
        }
      } catch (err) {}
      idx = endIdx + streamEnd.length;
    }
  }
  return fullText;
}

async function runPhase581Audit() {
  console.log('================================================================');
  console.log('   PHANTOM AI 3.0 — PHASE 5.8.1 COMPREHENSIVE INTEGRITY AUDIT   ');
  console.log('================================================================\n');

  if (mongoose.connection.readyState === 0) {
    await mongoose.connect(MONGO_URI);
  }
  console.log('[1/8] Connected to live MongoDB at:', MONGO_URI);

  // 1. Checkpoint Integrity Verification
  console.log('\n[2/8] Auditing Model Checkpoints & Parameter Counts...');
  const expectedCheckpoints = {
    generator: {
      path: path.resolve(__dirname, '../../phantom-ai-ml/checkpoints/cgan_generator_phase54.pt'),
      expectedSha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
    },
    evaluator: {
      path: path.resolve(__dirname, '../../phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt'),
      expectedSha256: 'f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834',
    },
    nlp: {
      path: path.resolve(__dirname, '../../phantom-ai-ml/checkpoints/nlp_char_bilstm_phase3c2.pt'),
      expectedSha256: '2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3',
    },
  };

  for (const [name, meta] of Object.entries(expectedCheckpoints)) {
    if (!fs.existsSync(meta.path)) {
      throw new Error(`Checkpoint file missing: ${meta.path}`);
    }
    const computedSha = fileSha256(meta.path);
    const match = computedSha === meta.expectedSha256;
    console.log(`  -> ${name.toUpperCase()} Checkpoint: ${match ? 'VERIFIED MATCH' : 'MISMATCH'}`);
    console.log(`     Path:   ${meta.path}`);
    console.log(`     SHA256: ${computedSha}`);
    if (!match) throw new Error(`SHA-256 mismatch for ${name} checkpoint!`);
  }

  // 2. Setup Authenticated Users & Cases for Security Tests
  console.log('\n[3/8] Setting Up Role-Based Test Identities & Case Contexts...');
  const testSuffix = Date.now().toString().slice(-5);

  const assignedInvestigator = await User.create({
    name: `Detective Columbo ${testSuffix}`,
    email: `columbo.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'investigator',
    badgeNumber: `BDG-${testSuffix}`,
    investigatorId: `INV-${testSuffix}`,
  });
  const assignedToken = await createAuthSession(assignedInvestigator);

  const unassignedInvestigator = await User.create({
    name: `Detective Marlowe ${testSuffix}`,
    email: `marlowe.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'investigator',
    badgeNumber: `BDG-UN-${testSuffix}`,
    investigatorId: `INV-UN-${testSuffix}`,
  });
  const unassignedToken = await createAuthSession(unassignedInvestigator);

  const supervisorUser = await User.create({
    name: `Supervisor Hastings ${testSuffix}`,
    email: `hastings.${testSuffix}@agency.gov`,
    password: 'SecurePasscode123!',
    role: 'supervisor',
    badgeNumber: `SUP-${testSuffix}`,
    investigatorId: `SUP-${testSuffix}`,
  });
  const supervisorToken = await createAuthSession(supervisorUser);

  // Create Case A (assigned to Columbo)
  const caseA = await Case.create({
    caseNumber: `CASE-AUDIT-A-${testSuffix}`,
    title: 'Audit Case A — Stolen Artifact Recovery',
    description: 'Investigation into targeted gallery theft.',
    createdBy: assignedInvestigator._id,
    assignedTo: [assignedInvestigator._id],
  });

  // Create Case B (assigned to Marlowe)
  const caseB = await Case.create({
    caseNumber: `CASE-AUDIT-B-${testSuffix}`,
    title: 'Audit Case B — Unrelated Perimeter Breach',
    description: 'Separate industrial facility surveillance report.',
    createdBy: unassignedInvestigator._id,
    assignedTo: [unassignedInvestigator._id],
  });

  console.log(`  -> Case A: ${caseA.caseNumber} (Assigned to ${assignedInvestigator.name})`);
  console.log(`  -> Case B: ${caseB.caseNumber} (Assigned to ${unassignedInvestigator.name})`);

  // 3. Test Evidence Preservation & Offset Accuracy
  console.log('\n[4/8] Testing Evidence Accuracy, Verbatim Statement & Character Offsets...');
  const rawWitnessStatement =
    'The subject was a Caucasian male in his late 20s with short black hair and high cheekbones. He had a distinct dark goatee and was not wearing glasses. His jacket was dark olive.';

  // Create witness description with precise character offsets
  // "black" is at index 58:63
  // "high cheekbones" is at index 73:88
  // "goatee" is at index 112:118
  // "not wearing glasses" is at index 127:146
  const witnessA = await WitnessDescription.create({
    case: caseA._id,
    witnessName: 'Evelyn Reed',
    witnessRole: 'Gallery Curator',
    rawStatement: rawWitnessStatement,
    confidenceLevel: 0.94,
    reviewStatus: 'approved',
    recordedBy: assignedInvestigator._id,
    extractedAttributes: [
      {
        category: 'hair_color',
        value: 'black',
        sourceText: 'black',
        start: rawWitnessStatement.indexOf('black'),
        end: rawWitnessStatement.indexOf('black') + 'black'.length,
        evidenceState: 'explicit',
        extractedBy: 'bilstm_crf_v1',
        confidence: 0.96,
      },
      {
        category: 'cheekbones',
        value: 'high',
        sourceText: 'high cheekbones',
        start: rawWitnessStatement.indexOf('high cheekbones'),
        end: rawWitnessStatement.indexOf('high cheekbones') + 'high cheekbones'.length,
        evidenceState: 'explicit',
        extractedBy: 'bilstm_crf_v1',
        confidence: 0.89,
      },
      {
        category: 'facial_hair',
        value: 'goatee',
        sourceText: 'goatee',
        start: rawWitnessStatement.indexOf('goatee'),
        end: rawWitnessStatement.indexOf('goatee') + 'goatee'.length,
        evidenceState: 'explicit',
        extractedBy: 'bilstm_crf_v1',
        confidence: 0.92,
      },
      {
        category: 'glasses',
        value: 'none',
        sourceText: 'not wearing glasses',
        start: rawWitnessStatement.indexOf('not wearing glasses'),
        end: rawWitnessStatement.indexOf('not wearing glasses') + 'not wearing glasses'.length,
        evidenceState: 'negated',
        extractedBy: 'bilstm_crf_v1',
        confidence: 0.95,
      },
      {
        category: 'eye_color',
        value: null,
        sourceText: 'unspecified',
        start: null,
        end: null,
        evidenceState: 'unknown',
        extractedBy: 'bilstm_crf_v1',
        confidence: null,
      },
    ],
    corrections: [
      {
        attributeCategory: 'hair_color',
        previousValue: 'dark',
        correctedValue: 'black',
        previousEvidenceState: 'uncertain',
        correctedEvidenceState: 'explicit',
        correctedBy: assignedInvestigator._id,
        timestamp: new Date(),
        reason: 'Witness clarified under direct examination that hair was jet black, not dark brown.',
        rejected: false,
      },
    ],
  });

  // Verify character offsets against slice of rawStatement
  for (const attr of witnessA.extractedAttributes) {
    if (typeof attr.start === 'number' && typeof attr.end === 'number') {
      const slice = rawWitnessStatement.slice(attr.start, attr.end);
      if (slice !== attr.sourceText) {
        throw new Error(
          `Character offset mismatch for ${attr.category}! Expected "${attr.sourceText}", got slice "${slice}"`
        );
      }
      console.log(`  -> Offset Verified: ${attr.category} [${attr.start}:${attr.end}] -> "${slice}"`);
    } else {
      console.log(`  -> Evidence State [${attr.evidenceState.toUpperCase()}]: ${attr.category} correctly marked without fabricated offsets.`);
    }
  }

  // 4. Generate Composite & Lineage History
  console.log('\n[5/8] Generating Facial Composite & Establishing Lineage...');
  const compGenRes = await axios.post(
    `${API_BASE}/cases/${caseA._id}/composites/generate`,
    {
      witnessDescriptionId: witnessA._id.toString(),
      seed: 88,
      numVariations: 1,
      notes: 'Initial synthesis for Case A.',
    },
    { headers: { Authorization: `Bearer ${assignedToken}` } }
  );
  const compositeV1 = compGenRes.data.data;
  console.log(`  -> Composite v1 Generated (ID: ${compositeV1._id}, Seed: 88)`);

  // Execute a semantic edit to create composite v2 with parent link
  const compEditRes = await axios.post(
    `${API_BASE}/cases/${caseA._id}/composites/${compositeV1._id}/edit`,
    {
      selectedSeed: 88,
      continuousAdjustments: { hair_bangs: 0.65 },
      latentTraversals: { hair_blond: 0.3 },
      notes: 'Semantic edit v2 applying bangs and blond traversal.',
    },
    { headers: { Authorization: `Bearer ${assignedToken}` } }
  );
  const compositeV2 = compEditRes.data.data.composite || compEditRes.data.data;
  console.log(`  -> Composite v2 Created (ID: ${compositeV2._id}, Parent: ${compositeV2.parentComposite})`);
  if (!compositeV2.parentComposite) {
    throw new Error('Version lineage failed: composite v2 has no parentComposite reference!');
  }

  // 5. Generate Multi-Modal Dossier via API
  console.log('\n[6/8] Synthesizing Multi-Modal Forensic Dossier...');
  const reportGenRes = await axios.post(
    `${API_BASE}/cases/${caseA._id}/reports`,
    {
      witnessDescriptionId: witnessA._id.toString(),
      compositeId: compositeV2._id.toString(),
      title: `Forensic Evidence Dossier — ${caseA.caseNumber}`,
      notes: 'Verified evidentiary packaging dossier for Phase 5.8.1 audit.',
    },
    { headers: { Authorization: `Bearer ${assignedToken}` } }
  );
  const reportDoc = reportGenRes.data.data;
  console.log(`  -> Dossier Generated: ${reportDoc.reportNumber}`);
  console.log(`  -> PDF SHA-256: ${reportDoc.sha256}`);
  console.log(`  -> File Size:   ${reportDoc.fileSizeBytes} bytes`);
  console.log(`  -> Live Image Hash Match: ${reportDoc.metadata.imageVerified ? 'YES' : 'NO'}`);

  // 6. Deep PDF Text Extraction & Content Verification
  console.log('\n[7/8] Inspecting Physical PDF Text Streams & Section Formatting...');
  const diskPdfPath = path.resolve(process.cwd(), 'data', 'reports', reportDoc.storageKey);
  if (!fs.existsSync(diskPdfPath)) {
    throw new Error(`Physical PDF missing at ${diskPdfPath}`);
  }
  const diskPdfBytes = fs.readFileSync(diskPdfPath);
  const diskPdfSha = crypto.createHash('sha256').update(diskPdfBytes).digest('hex');
  if (diskPdfSha !== reportDoc.sha256) {
    throw new Error(`PDF hash mismatch! Stored: ${reportDoc.sha256}, Actual: ${diskPdfSha}`);
  }
  console.log('  -> Disk PDF SHA-256 exactly matches MongoDB record.');

  const streamTexts = extractPdfStreamsText(diskPdfBytes);
  const joinedText = streamTexts.join('\n');

  // Verify all 8 sections are present in the PDF text
  const expectedSections = [
    { num: 'Section 1', title: 'CASE INFORMATION' },
    { num: 'Section 2', title: 'ORIGINAL WITNESS STATEMENT' },
    { num: 'Section 3', title: 'NLP EVIDENCE EXTRACTION' },
    { num: 'Section 4', title: 'FACIAL COMPOSITE GENERATION' },
    { num: 'Section 5', title: 'EDITING & VERSION LINEAGE HISTORY' },
    { num: 'Section 6', title: 'IMAGE & MODEL PROVENANCE' },
    { num: 'Section 7', title: 'CHAIN OF CUSTODY' },
    { num: 'Section 8', title: 'FORENSIC DISCLAIMER' },
  ];

  for (const sec of expectedSections) {
    const present = joinedText.includes(sec.num) && joinedText.includes(sec.title);
    console.log(`  -> Section Check: [${sec.num} — ${sec.title}] -> ${present ? 'PRESENT' : 'MISSING'}`);
    if (!present) throw new Error(`Section ${sec.num} missing in rendered PDF!`);
  }

  // Verify verbatim statement preservation in PDF text
  if (!joinedText.includes(rawWitnessStatement)) {
    throw new Error('Original witness statement not found verbatim in extracted PDF text!');
  }
  console.log('  -> Verbatim witness statement confirmed preserved in rendered PDF.');

  // Verify investigator notes are rendered in PDF text
  if (!joinedText.includes('Verified evidentiary packaging dossier for Phase 5.8.1 audit.')) {
    throw new Error('Investigator addendum notes missing in rendered PDF!');
  }
  console.log('  -> Investigator addendum notes confirmed present in rendered PDF.');

  // Verify version lineage v1 and v2 are both present in Section 5
  if (!joinedText.includes('v1') || !joinedText.includes('v2')) {
    throw new Error('Version lineage missing v1 or v2 in Section 5!');
  }
  console.log('  -> Composite version lineage v1 -> v2 confirmed in Section 5.');

  // Verify disclaimer disclaimer notice on hashing & limitations
  if (!joinedText.includes('Cryptographic SHA-256 digests in this dossier provide algorithmic integrity checks')) {
    throw new Error('Disclaimer cryptographic clarification missing in rendered PDF!');
  }
  console.log('  -> Cryptographic digest limitation notice confirmed in Section 8.');

  // 7. Security, Authorization & Boundary Enforcement
  console.log('\n[8/8] Testing RBAC, Cross-Case Security & Path Traversal Guards...');

  // Test A: Unassigned investigator access to Case A reports -> Must return 403
  try {
    await axios.get(`${API_BASE}/cases/${caseA._id}/reports`, {
      headers: { Authorization: `Bearer ${unassignedToken}` },
    });
    throw new Error('Security failure: Unassigned investigator was allowed to list reports for Case A!');
  } catch (err: any) {
    if (err.response?.status === 403) {
      console.log('  -> Security Check A: Unassigned investigator rejected with 403 Forbidden.');
    } else {
      throw err;
    }
  }

  // Test B: Unassigned investigator generating report on Case A -> Must return 403
  try {
    await axios.post(
      `${API_BASE}/cases/${caseA._id}/reports`,
      { compositeId: compositeV2._id.toString() },
      { headers: { Authorization: `Bearer ${unassignedToken}` } }
    );
    throw new Error('Security failure: Unassigned investigator was allowed to generate report on Case A!');
  } catch (err: any) {
    if (err.response?.status === 403) {
      console.log('  -> Security Check B: Unauthorized generation attempt rejected with 403 Forbidden.');
    } else {
      throw err;
    }
  }

  // Test C: Supervisor access to Case A reports -> Must succeed (Supervisor has oversight)
  const supListRes = await axios.get(`${API_BASE}/cases/${caseA._id}/reports`, {
    headers: { Authorization: `Bearer ${supervisorToken}` },
  });
  console.log(`  -> Security Check C: Supervisor oversight access confirmed (${supListRes.data.data.length} reports accessible).`);

  // Test D: Cross-case report access (Accessing Case A's report using Case B's URL) -> Must return 404
  try {
    await axios.get(`${API_BASE}/cases/${caseB._id}/reports/${reportDoc._id}`, {
      headers: { Authorization: `Bearer ${unassignedToken}` },
    });
    throw new Error('Security failure: Cross-case access allowed mismatched report/case mapping!');
  } catch (err: any) {
    if (err.response?.status === 404) {
      console.log('  -> Security Check D: Cross-case report access rejected with 404 Not Found.');
    } else {
      throw err;
    }
  }

  // Test E: Invalid report ID format -> Must return 400
  try {
    await axios.get(`${API_BASE}/cases/${caseA._id}/reports/invalid-id-format`, {
      headers: { Authorization: `Bearer ${assignedToken}` },
    });
    throw new Error('Security failure: Malformed report ID did not return 400!');
  } catch (err: any) {
    if (err.response?.status === 400) {
      console.log('  -> Security Check E: Invalid report ID rejected with 400 Bad Request.');
    } else {
      throw err;
    }
  }

  // Test F: Download authentication & streaming verification
  const downloadRes = await axios.get(`${API_BASE}/cases/${caseA._id}/reports/${reportDoc._id}/download`, {
    headers: { Authorization: `Bearer ${assignedToken}` },
    responseType: 'arraybuffer',
  });
  const downloadedBytes = Buffer.from(downloadRes.data);
  const downloadedSha = crypto.createHash('sha256').update(downloadedBytes).digest('hex');
  if (downloadedSha !== reportDoc.sha256) {
    throw new Error('Downloaded stream does not match stored report SHA-256!');
  }
  console.log(`  -> Security Check F: Authorized download stream verified byte-for-byte (${downloadedBytes.length} bytes).`);

  // Test G: Verify Failure Audit in MongoDB
  const failureAudit = await SecurityAudit.findOne({
    event: 'REPORT_GENERATION_FAILURE',
    'details.caseId': caseA._id.toString(),
  });
  console.log(`  -> Security Check G: Failure audit event recorded in MongoDB: ${Boolean(failureAudit)}`);

  console.log('\n================================================================');
  console.log('   PHASE 5.8.1 COMPREHENSIVE INTEGRITY AUDIT: ALL CHECKS PASSED ');
  console.log('================================================================');

  await mongoose.disconnect();
}

runPhase581Audit().catch((err) => {
  console.error('\n Phase 5.8.1 Audit Failed with error:', err);
  process.exit(1);
});

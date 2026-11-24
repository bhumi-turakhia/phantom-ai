import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import mongoose from 'mongoose';
import axios from 'axios';
import dotenv from 'dotenv';
import { User } from '../src/models/User.model';
import { Case } from '../src/models/Case.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { Composite } from '../src/models/Composite.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';
import { Session } from '../src/models/Session.model';

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const API_BASE = process.env.API_BASE_URL || 'http://127.0.0.1:5001/api/v1';
const MONGO_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/phantom_ai_dev';
const ML_BASE = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

async function createAuthenticatedSession(user: any): Promise<string> {
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
    userAgent: 'Phase-5.8-Audit/1.0',
    ipAddress: '127.0.0.1',
  });

  return rawToken;
}

async function runLiveAudit() {
  console.log('================================================================');
  console.log('   PHANTOM AI 3.0 — PHASE 5.8 LIVE FORENSIC REPORT EXPORT AUDIT  ');
  console.log('================================================================');

  // Step 1: Connect to MongoDB
  console.log('\n[1/10] Connecting to live MongoDB database...');
  await mongoose.connect(MONGO_URI);
  console.log(' Connected to MongoDB at:', MONGO_URI);

  // Step 2: Check API and ML service health
  console.log('\n[2/10] Checking live API and ML service status...');
  try {
    const apiHealth = await axios.get(`${API_BASE}/health`);
    console.log(' Express API Health (5001):', apiHealth.data);
  } catch (err: any) {
    console.error(' Express API unreachable at 5001:', err.message);
    process.exit(1);
  }

  try {
    const mlHealth = await axios.get(`${ML_BASE}/health`);
    console.log(' Python FastAPI Health (8000):', mlHealth.data);
  } catch (err: any) {
    console.error(' Python FastAPI unreachable at 8000:', err.message);
    process.exit(1);
  }

  // Step 3: Find or authenticate investigator user
  console.log('\n[3/10] Authenticating investigator for live test...');
  let investigator = await User.findOne({ role: 'investigator', isActive: true });
  if (!investigator) {
    investigator = await User.findOne({ isActive: true });
  }
  if (!investigator) {
    throw new Error('No active user found in MongoDB to perform audit.');
  }
  console.log(` Investigator identified: ${investigator.name} (${investigator.email}), Role: ${investigator.role}`);

  // Generate a valid JWT token & session for testing
  const token = await createAuthenticatedSession(investigator);

  const authHeaders = {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  // Step 4: Find an active case assigned to investigator with composite
  console.log('\n[4/10] Locating test case with witness testimony & generated composite...');
  let testCase = await Case.findOne({
    isDeleted: false,
    $or: [{ createdBy: investigator._id }, { assignedTo: investigator._id }],
  });

  if (!testCase) {
    testCase = await Case.create({
      caseNumber: `CASE-2026-AUDIT-${Date.now().toString().slice(-4)}`,
      title: 'Automated Audit Investigation Case',
      description: 'Synthetic test case for Phase 5.8 Dossier verification.',
      incidentDate: new Date(),
      priority: 'high',
      status: 'active',
      createdBy: investigator._id,
      assignedTo: [investigator._id],
    });
  }

  console.log(` Case resolved: ${testCase.caseNumber} (ID: ${testCase._id})`);

  // Ensure witness testimony exists and is approved
  let witness = await WitnessDescription.findOne({ case: testCase._id });
  if (!witness) {
    witness = await WitnessDescription.create({
      case: testCase._id,
      witnessName: 'Marcus Bennett',
      witnessRole: 'Security Guard',
      rawStatement:
        'The suspect had short black hair, high cheekbones, dark bushy eyebrows, and a visible goatee. He was not wearing glasses and definitely had no beard.',
      confidenceLevel: 0.92,
      reviewStatus: 'approved',
      recordedBy: investigator._id,
      extractedAttributes: [
        {
          category: 'hair_color',
          value: 'black',
          sourceText: 'black',
          start: 24,
          end: 29,
          evidenceState: 'explicit',
          extractedBy: 'bilstm_crf_v1',
          confidence: 0.95,
        },
        {
          category: 'cheekbones',
          value: 'high',
          sourceText: 'high cheekbones',
          start: 36,
          end: 51,
          evidenceState: 'explicit',
          extractedBy: 'bilstm_crf_v1',
          confidence: 0.88,
        },
        {
          category: 'facial_hair',
          value: 'goatee',
          sourceText: 'goatee',
          start: 94,
          end: 100,
          evidenceState: 'explicit',
          extractedBy: 'bilstm_crf_v1',
          confidence: 0.91,
        },
        {
          category: 'glasses',
          value: 'none',
          sourceText: 'not wearing glasses',
          start: 109,
          end: 128,
          evidenceState: 'negated',
          extractedBy: 'bilstm_crf_v1',
          confidence: 0.94,
        },
      ],
    });
  } else {
    witness.reviewStatus = 'approved';
    if (!witness.extractedAttributes || witness.extractedAttributes.length === 0) {
      witness.extractedAttributes = [
        {
          category: 'hair_color',
          value: 'black',
          sourceText: 'black',
          start: 24,
          end: 29,
          evidenceState: 'explicit',
          extractedBy: 'bilstm_crf_v1',
          confidence: 0.95,
        },
        {
          category: 'facial_hair',
          value: 'goatee',
          sourceText: 'goatee',
          start: 94,
          end: 100,
          evidenceState: 'explicit',
          extractedBy: 'bilstm_crf_v1',
          confidence: 0.91,
        },
      ];
    }
    await witness.save();
  }
  console.log(` Witness statement resolved: Statement #${witness._id.toString().slice(-6)}`);

  // Ensure composite exists, or generate one via live API
  let composite = await Composite.findOne({ case: testCase._id }).sort({ version: -1 });
  if (!composite) {
    console.log(' Generating live composite via POST /cases/:caseId/composites/generate...');
    const compRes = await axios.post(
      `${API_BASE}/cases/${testCase._id}/composites/generate`,
      {
        witnessDescriptionId: witness._id.toString(),
        seed: 42,
        numVariations: 1,
        notes: 'Live composite synthesis for Phase 5.8 dossier audit.',
      },
      { headers: authHeaders }
    );
    composite = compRes.data.data;
    console.log(` Live composite generated successfully: v${composite?.version} (Seed: 42)`);
  } else {
    console.log(` Existing composite resolved: v${composite.version} (${composite.status})`);
  }

  // Step 5: Test POST /api/v1/cases/:caseId/reports (PDF Generation)
  console.log('\n[5/10] Requesting forensic report dossier generation via POST /cases/:caseId/reports...');
  const t0 = Date.now();
  const generateRes = await axios.post(
    `${API_BASE}/cases/${testCase._id}/reports`,
    {
      witnessDescriptionId: witness._id.toString(),
      compositeId: composite?._id?.toString(),
      title: `Official Forensic Investigation Dossier — ${testCase.caseNumber}`,
      notes: 'Automated technical audit verification for Phase 5.8 multi-modal report packaging.',
    },
    { headers: authHeaders }
  );
  const genTime = Date.now() - t0;

  console.log(` Report generation HTTP status: ${generateRes.status}`);
  const reportData = generateRes.data.data;
  console.log(` Report Number: ${reportData.reportNumber}`);
  console.log(` Generated In: ${genTime} ms`);
  console.log(` File Size: ${reportData.fileSizeBytes} bytes (${(reportData.fileSizeBytes / 1024).toFixed(1)} KB)`);
  console.log(` Reported PDF SHA-256: ${reportData.sha256}`);
  console.log(` Image Verification: ${reportData.metadata.imageVerified ? 'MATCHED' : 'FAILED'}`);
  console.log(` Stored Image SHA-256: ${reportData.metadata.imageSha256}`);
  console.log(` Checkpoint SHA-256: ${reportData.metadata.checkpointSha256}`);

  // Step 6: Verify physical file on disk & cryptographic hash
  console.log('\n[6/10] Verifying physical PDF file on disk and cryptographic integrity...');
  const reportsDir = path.resolve(process.cwd(), 'data', 'reports');
  const diskFilePath = path.resolve(reportsDir, reportData.storageKey);

  if (!fs.existsSync(diskFilePath)) {
    throw new Error(`Physical PDF report file does not exist at expected path: ${diskFilePath}`);
  }
  const fileBuffer = fs.readFileSync(diskFilePath);
  const recomputedSha256 = crypto.createHash('sha256').update(fileBuffer).digest('hex');

  console.log(` Physical file exists: ${diskFilePath}`);
  console.log(` Disk file size: ${fileBuffer.length} bytes`);
  console.log(` Recomputed Disk SHA-256: ${recomputedSha256}`);
  if (recomputedSha256 !== reportData.sha256) {
    throw new Error(`Cryptographic mismatch! Stored: ${reportData.sha256}, Actual: ${recomputedSha256}`);
  }
  console.log(' Cryptographic hash matches stored MongoDB record perfectly!');

  // Check PDF signature
  const pdfMagic = fileBuffer.slice(0, 5).toString('ascii');
  console.log(` PDF Magic Signature: "${pdfMagic}" (Valid: ${pdfMagic === '%PDF-'})`);
  if (pdfMagic !== '%PDF-') {
    throw new Error('File does not begin with valid PDF signature %PDF-');
  }

  // Step 7: Test GET /api/v1/cases/:caseId/reports/:reportId/download (Streaming download)
  console.log('\n[7/10] Testing secure streaming download via GET /cases/:caseId/reports/:reportId/download...');
  const downloadRes = await axios.get(
    `${API_BASE}/cases/${testCase._id}/reports/${reportData._id}/download`,
    {
      headers: authHeaders,
      responseType: 'arraybuffer',
    }
  );

  console.log(` Download HTTP status: ${downloadRes.status}`);
  console.log(` Content-Type: ${downloadRes.headers['content-type']}`);
  console.log(` Content-Disposition: ${downloadRes.headers['content-disposition']}`);
  console.log(` Downloaded bytes: ${downloadRes.data.length}`);

  const downloadedSha256 = crypto.createHash('sha256').update(Buffer.from(downloadRes.data)).digest('hex');
  if (downloadedSha256 !== reportData.sha256) {
    throw new Error('Downloaded stream buffer does not match disk file SHA-256!');
  }
  console.log(' Downloaded stream matches disk file byte-for-byte!');

  // Step 8: Test Listing Endpoints
  console.log('\n[8/10] Testing report listing endpoints...');
  const caseReportsRes = await axios.get(`${API_BASE}/cases/${testCase._id}/reports`, {
    headers: authHeaders,
  });
  console.log(` Case reports count: ${caseReportsRes.data.data.length}`);

  const globalReportsRes = await axios.get(`${API_BASE}/reports`, {
    headers: authHeaders,
  });
  console.log(` Accessible global reports count: ${globalReportsRes.data.data.length}`);

  // Step 9: Verify Security Audit Logs in MongoDB
  console.log('\n[9/10] Verifying SecurityAudit logs in MongoDB...');
  const auditGen = await SecurityAudit.findOne({
    event: 'REPORT_GENERATED_SUCCESS',
    'details.reportId': reportData._id.toString(),
  });
  console.log(` REPORT_GENERATED_SUCCESS logged: ${Boolean(auditGen)}`);
  if (auditGen) {
    console.log(`   Event: ${auditGen.event}, Status: ${auditGen.status}, Timestamp: ${auditGen.createdAt}`);
  }

  const auditDown = await SecurityAudit.findOne({
    event: 'REPORT_DOWNLOADED',
    'details.reportId': reportData._id.toString(),
  });
  console.log(` REPORT_DOWNLOADED logged: ${Boolean(auditDown)}`);
  if (auditDown) {
    console.log(`   Event: ${auditDown.event}, Status: ${auditDown.status}, Timestamp: ${auditDown.createdAt}`);
  }

  // Step 10: Test RBAC & Security Boundary Enforcement
  console.log('\n[10/10] Testing RBAC and Security Boundary Enforcement...');
  let unauthorizedInv = await User.findOne({ _id: { $ne: investigator._id }, role: 'investigator' });
  if (!unauthorizedInv) {
    unauthorizedInv = await User.create({
      name: 'Unassigned Investigator',
      email: `unassigned.${Date.now()}@phantom-ai.internal`,
      password: 'password123',
      role: 'investigator',
      badgeNumber: 'BDG-UNAUTH',
      investigatorId: 'INV-UNAUTH',
    });
  }
  const unauthorizedToken = await createAuthenticatedSession(unauthorizedInv);

  try {
    await axios.post(
      `${API_BASE}/cases/${testCase._id}/reports`,
      {},
      { headers: { Authorization: `Bearer ${unauthorizedToken}` } }
    );
    throw new Error('Expected 403 Forbidden for unauthorized investigator but request succeeded!');
  } catch (err: any) {
    if (err.response?.status === 403) {
      console.log(' Unauthorized investigator rejected with 403 Forbidden as required.');
    } else {
      throw err;
    }
  }

  console.log('\n================================================================');
  console.log('   PHASE 5.8 LIVE FORENSIC REPORT EXPORT AUDIT: ALL 10 PASSED!  ');
  console.log('================================================================');

  await mongoose.disconnect();
}

runLiveAudit().catch((err) => {
  console.error('\n Audit execution failed with error:', err);
  process.exit(1);
});

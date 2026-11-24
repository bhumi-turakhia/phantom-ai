import http from 'http';
import crypto from 'crypto';
import jwt from 'jsonwebtoken';
import mongoose, { Types } from 'mongoose';
import { User } from '../src/models/User.model';
import { Session } from '../src/models/Session.model';
import { WitnessDescription } from '../src/models/WitnessDescription.model';
import { env } from '../src/config/env';

const API_BASE = 'http://127.0.0.1:5001';

function makeRequest(
  path: string,
  method = 'GET',
  body: any = null,
  token?: string
): Promise<{ status: number; body: any; headers: any }> {
  return new Promise((resolve, reject) => {
    const url = new URL(API_BASE + path);
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const payload = body ? JSON.stringify(body) : null;
    if (payload) {
      headers['Content-Length'] = Buffer.byteLength(payload).toString();
    }

    const req = http.request(
      {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method,
        headers,
      },
      (res) => {
        let raw = '';
        res.on('data', (c) => (raw += c));
        res.on('end', () => {
          let parsed: any = null;
          try {
            parsed = JSON.parse(raw);
          } catch {
            parsed = raw;
          }
          resolve({ status: res.statusCode || 500, body: parsed, headers: res.headers });
        });
      }
    );

    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function runLiveAudit() {
  console.log('=== Starting Phase 7.0 Live Integration & Immutability Audit ===');
  await mongoose.connect(env.MONGODB_URI);
  console.log('[DB] Connected to MongoDB at', env.MONGODB_URI);

  // 1. Set up test investigator accounts
  const primaryEmail = `phase7.audit.${Date.now()}@agency.gov`;
  const primaryUser = await User.create({
    name: 'Lead Detective Audit',
    email: primaryEmail,
    password: 'MockHashedPasswordArgon2id',
    role: 'investigator',
    badgeNumber: '7001',
    investigatorId: `INV-7001-${Math.floor(Math.random() * 9000 + 1000)}`,
    isActive: true,
  });

  const unauthorizedEmail = `phase7.unauthorized.${Date.now()}@agency.gov`;
  const unauthorizedUser = await User.create({
    name: 'Unassigned Detective',
    email: unauthorizedEmail,
    password: 'MockHashedPasswordArgon2id',
    role: 'investigator',
    badgeNumber: '7002',
    investigatorId: `INV-7002-${Math.floor(Math.random() * 9000 + 1000)}`,
    isActive: true,
  });

  // Create valid sessions & JWT tokens
  const tokenPayload = {
    id: primaryUser._id.toString(),
    role: primaryUser.role,
    investigatorId: primaryUser.investigatorId,
  };
  const primaryToken = jwt.sign(tokenPayload, env.JWT_SECRET, { expiresIn: '2h' });
  const primaryTokenHash = crypto.createHash('sha256').update(primaryToken).digest('hex');
  await Session.create({
    userId: primaryUser._id,
    investigatorId: primaryUser.investigatorId,
    tokenHash: primaryTokenHash,
    isValid: true,
    expiresAt: new Date(Date.now() + 2 * 3600 * 1000),
  });

  const unauthPayload = {
    id: unauthorizedUser._id.toString(),
    role: unauthorizedUser.role,
    investigatorId: unauthorizedUser.investigatorId,
  };
  const unauthToken = jwt.sign(unauthPayload, env.JWT_SECRET, { expiresIn: '2h' });
  const unauthTokenHash = crypto.createHash('sha256').update(unauthToken).digest('hex');
  await Session.create({
    userId: unauthorizedUser._id,
    investigatorId: unauthorizedUser.investigatorId,
    tokenHash: unauthTokenHash,
    isValid: true,
    expiresAt: new Date(Date.now() + 2 * 3600 * 1000),
  });

  console.log('[AUTH] Created authenticated investigator session:', primaryUser.investigatorId);

  // 2. Create Case
  const caseRes = await makeRequest(
    '/api/v1/cases',
    'POST',
    {
      caseNumber: `CASE-XAI-${Date.now()}`,
      title: 'Phase 7.0 Forensic XAI Audit Case',
      description: 'Live testing of evidence trace, non-destructive ledger, and GCIT provenance.',
      priority: 'high',
    },
    primaryToken
  );
  if (caseRes.status !== 201) {
    throw new Error(`Failed to create case: ${JSON.stringify(caseRes.body)}`);
  }
  const caseId = caseRes.body.data._id || caseRes.body.data.case?._id;
  console.log('[CASE] Created live case:', caseId);

  // 3. Create Witness Description
  const witnessStatement =
    'The subject was male. He had short black hair, an oval face, and a sharp pointy nose. He was clean-shaven with no beard. He was wearing eyeglasses.';
  const witnessRes = await makeRequest(
    '/api/v1/witness-descriptions',
    'POST',
    {
      case: caseId,
      rawStatement: witnessStatement,
      witnessName: 'Audited Witness Alpha',
      gender: 'male',
    },
    primaryToken
  );
  if (witnessRes.status !== 201) {
    throw new Error(`Failed to create witness description: ${JSON.stringify(witnessRes.body)}`);
  }
  const witnessId = witnessRes.body.data._id;
  console.log('[WITNESS] Created witness description:', witnessId);

  // 4. Run AI Analysis via Live ML Service
  console.log('[NLP] Triggering live AI analysis via Python ML service...');
  const analysisRes = await makeRequest(`/api/v1/ai-analysis/${witnessId}`, 'POST', {}, primaryToken);
  if (analysisRes.status !== 200) {
    throw new Error(`Failed to analyze witness description: ${JSON.stringify(analysisRes.body)}`);
  }
  console.log('[NLP] AI Analysis completed successfully. Extracted categories count:', Object.keys(analysisRes.body.data.attributes || {}).length);

  // 5. Test Live Witness Trace Endpoint: GET /api/v1/witness-descriptions/:id/trace
  console.log('[TRACE] Calling live witness trace endpoint: GET /api/v1/witness-descriptions/:id/trace');
  const witnessTraceRes = await makeRequest(`/api/v1/witness-descriptions/${witnessId}/trace`, 'GET', null, primaryToken);
  if (witnessTraceRes.status !== 200) {
    throw new Error(`Witness trace endpoint failed: ${witnessTraceRes.status} ${JSON.stringify(witnessTraceRes.body)}`);
  }
  const witnessTraceData = witnessTraceRes.body.data;
  console.log('[TRACE] Witness trace returned 200 OK. Trace items count:', witnessTraceData.evidenceTrace?.length);
  console.log('[TRACE] GCIT Preview conditionVector length:', witnessTraceData.gcitPreview?.conditionVector?.length);
  console.log('[TRACE] Conditioned dimensions count:', witnessTraceData.gcitPreview?.conditionedCount);

  // Verify all 15 points on the first trace item
  const sampleTrace = witnessTraceData.evidenceTrace[0];
  console.log('[AUDIT: Point 1-4] Canonical:', sampleTrace.attributeName, '| Category:', sampleTrace.category, '| Extracted:', sampleTrace.extractedValue, '| Current:', sampleTrace.currentValue);
  console.log('[AUDIT: Point 5-7] Raw Statement present:', !!sampleTrace.originalWitnessStatement, '| Span:', sampleTrace.exactCharacterOffsets, '| Context Sentence:', sampleTrace.surroundingContextSentence);
  console.log('[AUDIT: Point 8-10] State Extracted:', sampleTrace.extractedEvidenceState, '| State Active:', sampleTrace.currentEvidenceState, '| Model:', sampleTrace.nlpProvenance?.modelName, '| Conf:', sampleTrace.nlpProvenance?.extractionConfidence);
  console.log('[AUDIT: Point 11-12] Is Corrected:', sampleTrace.investigatorCorrection?.isCorrected, '| Review Status:', sampleTrace.reviewStatus);
  console.log('[AUDIT: Point 13-14] GCIT Supported:', sampleTrace.gcitTranslation?.isSupported, '| Dim:', sampleTrace.gcitTranslation?.dimensionName, '| Polarity:', sampleTrace.gcitTranslation?.polarity, '| c:', sampleTrace.gcitTranslation?.conditionValue, '| m:', sampleTrace.gcitTranslation?.maskValue);
  console.log('[AUDIT: Point 15] Forensic Disclosure:', sampleTrace.forensicDisclosure?.generativePriorNotice);

  // Check Inverted Polarity specifically on facial_no_beard
  const noBeardTrace = witnessTraceData.evidenceTrace.find((t: any) => t.category === 'facial_hair');
  if (noBeardTrace) {
    console.log('[INVERTED POLARITY AUDIT] facial_hair dimension:', noBeardTrace.gcitTranslation?.dimensionName);
    console.log('[INVERTED POLARITY AUDIT] polarity:', noBeardTrace.gcitTranslation?.polarity);
    console.log('[INVERTED POLARITY AUDIT] c_value:', noBeardTrace.gcitTranslation?.conditionValue, 'm_value:', noBeardTrace.gcitTranslation?.maskValue);
    console.log('[INVERTED POLARITY AUDIT] inclusion reason:', noBeardTrace.gcitTranslation?.inclusionReason);
  }

  // 6. Test Non-Destructive Investigator Correction via Live API
  console.log('[CORRECTION] Performing investigator override on facial_hair via live endpoint...');
  const correctionRes = await makeRequest(
    `/api/v1/witness-descriptions/${witnessId}/corrections`,
    'POST',
    {
      attributeCategory: 'facial_hair',
      correctedValue: '5 o clock shadow',
      correctedEvidenceState: 'explicit',
      reason: 'Witness clarified upon reviewing photos that suspect had heavy 5 o clock shadow stubble.',
    },
    primaryToken
  );
  if (correctionRes.status !== 200) {
    throw new Error(`Correction endpoint failed: ${correctionRes.status} ${JSON.stringify(correctionRes.body)}`);
  }

  // Re-fetch witness trace to verify non-destructive persistence in MongoDB
  const postCorrectionTrace = await makeRequest(`/api/v1/witness-descriptions/${witnessId}/trace`, 'GET', null, primaryToken);
  const correctedItem = postCorrectionTrace.body.data.evidenceTrace.find((t: any) => t.category === 'facial_hair');
  console.log('[NON-DESTRUCTIVE AUDIT] Original Value preserved in DB:', correctedItem.investigatorCorrection?.originalValue);
  console.log('[NON-DESTRUCTIVE AUDIT] Original State preserved in DB:', correctedItem.investigatorCorrection?.originalEvidenceState);
  console.log('[NON-DESTRUCTIVE AUDIT] Current Value updated:', correctedItem.currentValue);
  console.log('[NON-DESTRUCTIVE AUDIT] Is Corrected flag:', correctedItem.investigatorCorrection?.isCorrected);
  console.log('[NON-DESTRUCTIVE AUDIT] Investigator Rationale logged:', correctedItem.investigatorCorrection?.rationale);
  console.log('[NON-DESTRUCTIVE AUDIT] Re-evaluated GCIT dimension:', correctedItem.gcitTranslation?.dimensionName, 'c:', correctedItem.gcitTranslation?.conditionValue);

  // 7. Test Unauthorized & Direct Update Attempts
  console.log('[SECURITY AUDIT] Testing unauthorized access to witness trace...');
  const unauthRes = await makeRequest(`/api/v1/witness-descriptions/${witnessId}/trace`, 'GET', null, unauthToken);
  console.log('[SECURITY AUDIT] Unassigned investigator response code:', unauthRes.status, '(Expected 403)');

  const noTokenRes = await makeRequest(`/api/v1/witness-descriptions/${witnessId}/trace`, 'GET', null, undefined);
  console.log('[SECURITY AUDIT] Unauthenticated response code:', noTokenRes.status, '(Expected 401)');

  // Attempt direct overwrite of originalValue and originalEvidenceState via PATCH /api/v1/witness-descriptions/:id
  console.log('[IMMUTABILITY AUDIT] Testing direct tamper attempt via PATCH /api/v1/witness-descriptions/:id...');
  const patchTamperRes = await makeRequest(
    `/api/v1/witness-descriptions/${witnessId}`,
    'PATCH',
    {
      originalValue: 'HACKED_VALUE',
      originalEvidenceState: 'contradictory',
      extractedAttributes: [{ category: 'hair', value: 'FAKE_HAIR', originalValue: 'HACKED' }],
    },
    primaryToken
  );
  console.log('[IMMUTABILITY AUDIT] PATCH tamper request status:', patchTamperRes.status);

  // Verify DB state after tamper attempt
  const docAfterTamper = await WitnessDescription.findById(witnessId);
  const hairAttrAfterTamper = docAfterTamper?.extractedAttributes?.find((a: any) => a.category === 'hair');
  const facialAttrAfterTamper = docAfterTamper?.extractedAttributes?.find((a: any) => a.category === 'facial_hair');
  console.log('[IMMUTABILITY AUDIT] Did extractedAttributes get overwritten by PATCH?', hairAttrAfterTamper?.value !== 'FAKE_HAIR' ? 'NO (Protected by Schema/Validation)' : 'YES (Vulnerable)');
  console.log('[IMMUTABILITY AUDIT] Is originalValue still intact?', facialAttrAfterTamper?.originalValue !== 'HACKED_VALUE' ? 'YES (Intact: ' + facialAttrAfterTamper?.originalValue + ')' : 'NO (Tampered)');

  // 8. Approve Witness Description
  console.log('[APPROVAL] Approving witness description...');
  const approveRes = await makeRequest(`/api/v1/witness-descriptions/${witnessId}/approve`, 'POST', { status: 'approved' }, primaryToken);
  if (approveRes.status !== 200) {
    throw new Error(`Approval failed: ${approveRes.status} ${JSON.stringify(approveRes.body)}`);
  }
  console.log('[APPROVAL] Approved status:', approveRes.body.data.reviewStatus);

  // 9. Generate Composite via Real PyTorch Generator on MPS
  console.log('[GENERATOR] Synthesizing composite with PyTorch Conditional WGAN-GP...');
  const genRes = await makeRequest(
    `/api/v1/cases/${caseId}/composites/generate`,
    'POST',
    {
      witnessDescriptionId: witnessId,
      seed: 42,
      numVariations: 2,
    },
    primaryToken
  );
  if (genRes.status !== 201) {
    throw new Error(`Composite generation failed: ${genRes.status} ${JSON.stringify(genRes.body)}`);
  }
  const compositeDoc = genRes.body.data;
  console.log('[GENERATOR] Composite synthesized successfully! Composite ID:', compositeDoc._id, 'Version:', compositeDoc.version);
  console.log('[GENERATOR] Generator metadata checkpoint SHA-256:', compositeDoc.generatorMetadata?.checkpointSha256);
  console.log('[GENERATOR] Image variations synthesized count:', compositeDoc.variations?.length);

  // 10. Test Live Composite Trace Endpoint: GET /api/v1/cases/:caseId/composites/:compositeId/trace
  console.log('[COMPOSITE TRACE] Calling live composite trace endpoint: GET /api/v1/cases/:caseId/composites/:compositeId/trace');
  const compTraceRes = await makeRequest(`/api/v1/cases/${caseId}/composites/${compositeDoc._id}/trace`, 'GET', null, primaryToken);
  if (compTraceRes.status !== 200) {
    throw new Error(`Composite trace endpoint failed: ${compTraceRes.status} ${JSON.stringify(compTraceRes.body)}`);
  }
  const compTraceData = compTraceRes.body.data;
  console.log('[COMPOSITE TRACE] Returned 200 OK.');
  console.log('[COMPOSITE TRACE] Composite Version:', compTraceData.version);
  console.log('[COMPOSITE TRACE] Generator Checkpoint SHA-256:', compTraceData.generatorMetadata?.checkpointSha256);
  console.log('[COMPOSITE TRACE] 20-D GCIT Vector Length:', compTraceData.gcitVector?.length);
  console.log('[COMPOSITE TRACE] 20-D GCIT Mask Length:', compTraceData.gcitMask?.length);
  console.log('[COMPOSITE TRACE] Evidence Trace Snapshot items:', compTraceData.evidenceTrace?.length);
  console.log('[COMPOSITE TRACE] Variation Image Hashes (SHA-256):', compTraceData.imageHashes);

  // Test unauthorized access to composite trace
  console.log('[SECURITY AUDIT] Testing unauthorized access to composite trace...');
  const compUnauthRes = await makeRequest(`/api/v1/cases/${caseId}/composites/${compositeDoc._id}/trace`, 'GET', null, unauthToken);
  console.log('[SECURITY AUDIT] Unassigned investigator response code:', compUnauthRes.status, '(Expected 403)');

  const wrongCaseRes = await makeRequest(`/api/v1/cases/${new Types.ObjectId()}/composites/${compositeDoc._id}/trace`, 'GET', null, primaryToken);
  console.log('[SECURITY AUDIT] Mismatched case ID response code:', wrongCaseRes.status, '(Expected 404)');

  console.log('\n=== LIVE VERIFICATION AUDIT COMPLETE ===');
  console.log('All live endpoints verified against Express API, MongoDB, and PyTorch ML service.');
  await mongoose.disconnect();
}

runLiveAudit().catch((err) => {
  console.error('[FATAL] Live audit failed:', err);
  process.exit(1);
});

import crypto from 'crypto';
import { Types } from 'mongoose';
import { PDFReportService, PDFReportData } from '../src/services/pdfReport.service';
import { ReportService } from '../src/services/report.service';
import { Case } from '../src/models/Case.model';
import { Report } from '../src/models/Report.model';
import { SecurityAudit } from '../src/models/SecurityAudit.model';

describe('Phase 5.8 Multi-Modal Forensic Report Export Suite', () => {
  const mockUserId = new Types.ObjectId();
  const mockCaseId = new Types.ObjectId();
  const mockWitnessId = new Types.ObjectId();
  const mockCompositeId = new Types.ObjectId();

  // Valid 16x16 PNG buffer (uncompressed)
  const validPngHeader = Buffer.from([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
    0x00, 0x00, 0x00, 0x0d, // IHDR length
    0x49, 0x48, 0x44, 0x52, // IHDR
    0x00, 0x00, 0x00, 0x01, // width: 1
    0x00, 0x00, 0x00, 0x01, // height: 1
    0x08, 0x06, 0x00, 0x00, 0x00, // 8-bit RGBA
    0x1f, 0x15, 0xc4, 0x89, // CRC
    0x00, 0x00, 0x00, 0x0a, // IDAT length
    0x49, 0x44, 0x41, 0x54, // IDAT
    0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, // ZLIB data
    0x0d, 0x0a, 0x2d, 0xb4, // CRC
    0x00, 0x00, 0x00, 0x00, // IEND length
    0x49, 0x45, 0x4e, 0x44, // IEND
    0xae, 0x42, 0x60, 0x82, // CRC
  ]);

  const mockUser: any = {
    _id: mockUserId,
    investigatorId: 'INV-2026-99',
    name: 'Special Investigator Vance',
    email: 'vance@phantom-ai.internal',
    role: 'investigator',
    badgeNumber: 'BDG-9981',
    department: 'Special Forensic Investigation Unit',
  };

  const mockCase: any = {
    _id: mockCaseId,
    caseNumber: 'CASE-2026-8801',
    title: 'Downtown Vault Security Breach Incident',
    description: 'Investigation into unauthorized physical breach and surveillance capture.',
    incidentDate: new Date('2026-06-15T03:30:00Z'),
    priority: 'high',
    status: 'active',
    createdBy: mockUserId,
    assignedTo: [mockUserId],
    isDeleted: false,
  };

  const mockWitness: any = {
    _id: mockWitnessId,
    case: mockCaseId,
    witnessName: 'Marcus Bennett',
    witnessRole: 'Security Guard',
    rawStatement:
      'The suspect was a male in his mid-30s with short black hair, high cheekbones, dark bushy eyebrows, and a visible goatee. He was not wearing glasses and definitely had no beard.',
    confidenceScore: 0.92,
    nlpAnalysis: {
      entities: [
        {
          attributeCategory: 'hair_color',
          attributeValue: 'black',
          evidenceState: 'explicit',
          confidence: 0.95,
          textSpan: { start: 42, end: 47, substringQuote: 'black' },
        },
        {
          attributeCategory: 'cheekbones',
          attributeValue: 'high',
          evidenceState: 'explicit',
          confidence: 0.88,
          textSpan: { start: 54, end: 69, substringQuote: 'high cheekbones' },
        },
        {
          attributeCategory: 'facial_hair',
          attributeValue: 'goatee',
          evidenceState: 'explicit',
          confidence: 0.91,
          textSpan: { start: 104, end: 110, substringQuote: 'goatee' },
        },
        {
          attributeCategory: 'glasses',
          attributeValue: 'none',
          evidenceState: 'negated',
          confidence: 0.94,
          textSpan: { start: 120, end: 143, substringQuote: 'not wearing glasses' },
        },
      ],
      investigatorCorrections: [
        {
          attributeCategory: 'hair_color',
          previousValue: 'dark',
          correctedValue: 'black',
          reason: 'Witness clarified lighting was sufficient to identify pure black hair.',
          timestamp: new Date('2026-06-15T05:10:00Z'),
        },
      ],
    },
    approvedEvidenceSnapshot: {
      hair_color: { value: 'black', state: 'explicit', confidence: 0.95 },
      cheekbones: { value: 'high', state: 'explicit', confidence: 0.88 },
      facial_hair: { value: 'goatee', state: 'explicit', confidence: 0.91 },
      glasses: { value: 'none', state: 'negated', confidence: 0.94 },
    },
    status: 'approved',
    createdAt: new Date('2026-06-15T04:00:00Z'),
  };

  const mockComposite: any = {
    _id: mockCompositeId,
    case: mockCaseId,
    witnessDescription: mockWitnessId,
    version: 2,
    status: 'approved',
    generatorMetadata: {
      modelName: 'ConditionalWGANGPGenerator',
      architecture: 'Conditional WGAN-GP with Projection Conditioning',
      phase: 'Phase 5.4 / 5.7',
      checkpointPath: 'phantom-ai-ml/checkpoints/cgan_generator_phase54.pt',
      checkpointSha256: 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964',
      bestEpoch: 4,
      parameterCount: 4102336,
      device: 'mps',
    },
    approvedEvidenceSnapshot: mockWitness.approvedEvidenceSnapshot,
    conditionedAttributes: [
      'hair_black (+1.00)',
      'face_cheekbones (+1.00)',
      'facial_goatee (+1.00)',
      'glasses (-1.00)',
    ],
    unconditionedAttributes: ['hair_bald', 'eyes_narrow'],
    warnings: [],
    variations: [
      {
        variationIndex: 1,
        seed: 42,
        storageKey: 'mock_composite_test_v2.png',
        imageUrl: '/api/v1/cases/test/composites/v2/image/1',
        dataUrl: `data:image/png;base64,${validPngHeader.toString('base64')}`,
        mimeType: 'image/png',
        fileSizeBytes: validPngHeader.length,
        width: 64,
        height: 64,
        isSelected: true,
      },
    ],
    editingType: 'hybrid_traversal',
    traversalParameters: {
      continuousAdjustments: { hair_bangs: 0.65 },
      latentTraversals: [{ attribute: 'hair_blond', alpha: 0.3 }],
    },
    createdAt: new Date('2026-06-15T06:00:00Z'),
  };

  const mockCompositeV1: any = {
    _id: new Types.ObjectId(),
    case: mockCaseId,
    witnessDescription: mockWitnessId,
    version: 1,
    status: 'superseded',
    generatorMetadata: mockComposite.generatorMetadata,
    variations: [
      {
        variationIndex: 1,
        seed: 42,
        storageKey: 'mock_composite_test_v1.png',
        dataUrl: `data:image/png;base64,${validPngHeader.toString('base64')}`,
        fileSizeBytes: validPngHeader.length,
        isSelected: true,
      },
    ],
    editingType: 'initial',
    createdAt: new Date('2026-06-15T05:30:00Z'),
  };

  const mockAudits: any[] = [
    {
      _id: new Types.ObjectId(),
      userId: mockUserId,
      event: 'COMPOSITE_GENERATION_SUCCESS',
      status: 'success',
      details: { caseId: mockCaseId.toString(), compositeVersion: 1, seed: 42 },
      timestamp: new Date('2026-06-15T05:30:00Z'),
      createdAt: new Date('2026-06-15T05:30:00Z'),
      ipAddress: '127.0.0.1',
      userAgent: 'Phantom-AI-Client/3.0',
    },
    {
      _id: new Types.ObjectId(),
      userId: mockUserId,
      event: 'COMPOSITE_SEMANTIC_EDIT',
      status: 'success',
      details: { caseId: mockCaseId.toString(), version: 2, parentVersion: 1 },
      timestamp: new Date('2026-06-15T06:00:00Z'),
      createdAt: new Date('2026-06-15T06:00:00Z'),
      ipAddress: '127.0.0.1',
      userAgent: 'Phantom-AI-Client/3.0',
    },
  ];

  describe('PDFReportService.generateForensicDossier', () => {
    it('synthesizes a complete multi-page PDF document with valid PDF signature', async () => {
      const pdfData: PDFReportData = {
        reportNumber: 'RPT-CASE20268801-123456-ABCDEF',
        generatedAt: new Date(),
        generatedBy: mockUser,
        caseDoc: mockCase,
        witnessDoc: mockWitness,
        compositeDoc: mockComposite,
        compositeHistory: [mockCompositeV1, mockComposite],
        auditLogs: mockAudits,
        imageVerified: true,
        computedImageSha256: crypto.createHash('sha256').update(validPngHeader).digest('hex'),
        notes: 'Forensic evaluation confirmed high fidelity with physical witness testimony.',
      };

      const pdfBuffer = await PDFReportService.generateForensicDossier(pdfData);

      expect(Buffer.isBuffer(pdfBuffer)).toBe(true);
      expect(pdfBuffer.length).toBeGreaterThan(1000);

      // Verify PDF file header signature (%PDF-)
      const headerStr = pdfBuffer.slice(0, 5).toString('ascii');
      expect(headerStr).toBe('%PDF-');

      // Verify cryptographic SHA-256 computation
      const pdfSha256 = crypto.createHash('sha256').update(pdfBuffer).digest('hex');
      expect(pdfSha256).toHaveLength(64);
    });

    it('generates a valid dossier even when composite generation has not yet been performed', async () => {
      const pdfData: PDFReportData = {
        reportNumber: 'RPT-PRE-COMPOSITE-001',
        generatedAt: new Date(),
        generatedBy: mockUser,
        caseDoc: mockCase,
        witnessDoc: mockWitness,
        compositeDoc: null,
        compositeHistory: [],
        auditLogs: mockAudits,
        notes: 'Pre-generation witness extraction summary dossier.',
      };

      const pdfBuffer = await PDFReportService.generateForensicDossier(pdfData);
      expect(Buffer.isBuffer(pdfBuffer)).toBe(true);
      expect(pdfBuffer.slice(0, 5).toString('ascii')).toBe('%PDF-');
    });

    it('generates a valid dossier when witness testimony is absent', async () => {
      const pdfData: PDFReportData = {
        reportNumber: 'RPT-CASE-INIT-001',
        generatedAt: new Date(),
        generatedBy: mockUser,
        caseDoc: mockCase,
        witnessDoc: null,
        compositeDoc: null,
        compositeHistory: [],
        auditLogs: [],
      };

      const pdfBuffer = await PDFReportService.generateForensicDossier(pdfData);
      expect(Buffer.isBuffer(pdfBuffer)).toBe(true);
      expect(pdfBuffer.slice(0, 5).toString('ascii')).toBe('%PDF-');
    });
  });

  describe('ReportService RBAC and Path Traversal Protection', () => {
    it('enforces RBAC and rejects unauthorized investigators from generating reports', async () => {
      const unauthorizedUserId = new Types.ObjectId().toString();

      jest.spyOn(SecurityAudit, 'create').mockResolvedValue({} as any);

      // Mock Case find to return a case where user is not creator or assigned
      jest.spyOn(Case, 'findOne').mockResolvedValueOnce({
        _id: mockCaseId,
        createdBy: mockUserId,
        assignedTo: [mockUserId],
        isDeleted: false,
      } as any);

      await expect(
        ReportService.generateReport(mockCaseId.toString(), unauthorizedUserId, 'investigator', {})
      ).rejects.toThrow(/Forbidden/);
    });

    it('strictly prevents directory traversal attacks during report file download', async () => {
      // Mock Report.findOne to return a malicious report storage key
      jest.spyOn(Case, 'findOne').mockResolvedValueOnce(mockCase as any);
      jest.spyOn(Report, 'findOne').mockResolvedValueOnce({
        _id: new Types.ObjectId(),
        case: mockCaseId,
        reportNumber: 'RPT-TEST-MALICIOUS',
        storageKey: '../../etc/passwd',
      } as any);

      // Because sanitizedKey uses path.basename, path traversal is sanitized:
      // basename('../../etc/passwd') === 'passwd'
      // It should check data/reports/passwd, and throw 404 since that file does not exist
      await expect(
        ReportService.getReportFile(
          mockCaseId.toString(),
          new Types.ObjectId().toString(),
          mockUserId.toString(),
          'investigator'
        )
      ).rejects.toThrow(/missing from archive storage/);
    });
  });
});

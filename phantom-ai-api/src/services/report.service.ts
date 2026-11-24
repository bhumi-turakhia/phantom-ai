import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { Types } from 'mongoose';
import { Report, IReport } from '../models/Report.model';
import { Case } from '../models/Case.model';
import { User } from '../models/User.model';
import { WitnessDescription, IWitnessDescription } from '../models/WitnessDescription.model';
import { Composite, IComposite } from '../models/Composite.model';
import { SecurityAudit } from '../models/SecurityAudit.model';
import { ApiError } from '../utils/apiResponse';
import { assertCaseAccess } from './witnessDescription.service';
import { PDFReportService, PDFReportData } from './pdfReport.service';
import { logger } from '../config/logger';

const REPORTS_STORAGE_DIR = path.resolve(process.cwd(), 'data', 'reports');
const COMPOSITES_STORAGE_DIR = path.resolve(process.cwd(), 'data', 'composites');

const ensureReportsDir = (): void => {
  if (!fs.existsSync(REPORTS_STORAGE_DIR)) {
    fs.mkdirSync(REPORTS_STORAGE_DIR, { recursive: true });
  }
};

const recordAudit = async (
  event: any,
  status: 'success' | 'failure' | 'warning',
  userId: string,
  details: Record<string, unknown>,
  ipAddress?: string,
  userAgent?: string
): Promise<void> => {
  try {
    await SecurityAudit.create({
      userId: new Types.ObjectId(userId),
      event,
      status,
      details,
      ipAddress: ipAddress || '127.0.0.1',
      userAgent: userAgent || 'Phantom-AI-Dossier-Engine/3.0',
    });
  } catch (err) {
    logger.error('Failed to record security audit log for report:', err);
  }
};

export interface GenerateReportServiceInput {
  witnessDescriptionId?: string;
  compositeId?: string;
  compositeVersion?: number;
  title?: string;
  notes?: string;
  clientIp?: string;
  userAgent?: string;
}

export class ReportService {
  /**
   * Generates a complete forensic investigation PDF report and saves metadata
   */
  public static async generateReport(
    caseId: string,
    userId: string,
    userRole: string,
    input: GenerateReportServiceInput
  ): Promise<IReport> {
    ensureReportsDir();

    try {
      // 1. Authorize case access
      await assertCaseAccess(caseId, userId, userRole);

      // 2. Fetch Case document
      const caseDoc = await Case.findOne({ _id: caseId, isDeleted: false });
      if (!caseDoc) {
        throw new ApiError('Case file not found or has been deleted', 404);
      }

      // 3. Fetch User document (Investigator)
      const userDoc = await User.findById(userId);
      if (!userDoc) {
        throw new ApiError('Investigator user record not found', 404);
      }
      // 4. Resolve Witness Description
      let witnessDoc: IWitnessDescription | null = null;
      if (input.witnessDescriptionId) {
        witnessDoc = await WitnessDescription.findOne({
          _id: input.witnessDescriptionId,
          case: caseId,
        });
        if (!witnessDoc) {
          throw new ApiError('Specified witness description not found for this case', 404);
        }
      } else {
        // Fetch latest witness description for this case
        witnessDoc = await WitnessDescription.findOne({ case: caseId }).sort({ createdAt: -1 });
      }

      // 5. Resolve Composite
      let compositeDoc: IComposite | null = null;
      if (input.compositeId) {
        compositeDoc = await Composite.findOne({
          _id: input.compositeId,
          case: caseId,
        });
        if (!compositeDoc) {
          throw new ApiError('Specified composite not found for this case', 404);
        }
      } else if (input.compositeVersion) {
        compositeDoc = await Composite.findOne({
          case: caseId,
          version: input.compositeVersion,
        });
        if (!compositeDoc) {
          throw new ApiError(
            `Composite version ${input.compositeVersion} not found for this case`,
            404
          );
        }
      } else {
        // Fetch latest approved or newest composite for this case
        compositeDoc = await Composite.findOne({
          case: caseId,
          status: 'approved',
        }).sort({ version: -1 });

        if (!compositeDoc) {
          compositeDoc = await Composite.findOne({ case: caseId }).sort({ version: -1 });
        }
      }

      // 6. Trace Composite Lineage History if composite exists
      let compositeHistory: IComposite[] = [];
      let imageVerified = false;
      let computedImageSha256: string | undefined;

      if (compositeDoc) {
        compositeHistory = await Composite.find({
          case: caseId,
          version: { $lte: compositeDoc.version },
        }).sort({ version: 1 });

        // 7. Verify Lossless Image Integrity Live
        const variation =
          compositeDoc.variations.find((v) => v.isSelected) || compositeDoc.variations[0];

        if (variation) {
          let imageBuffer: Buffer | null = null;

          if (variation.storageKey) {
            const diskPath = path.isAbsolute(variation.storageKey)
              ? variation.storageKey
              : path.resolve(COMPOSITES_STORAGE_DIR, variation.storageKey);

            if (fs.existsSync(diskPath)) {
              imageBuffer = fs.readFileSync(diskPath);
            }
          }

          if (!imageBuffer && variation.dataUrl) {
            const base64Data = variation.dataUrl.replace(/^data:image\/\w+;base64,/, '');
            imageBuffer = Buffer.from(base64Data, 'base64');
          }

          if (imageBuffer && imageBuffer.length >= 8) {
            // Verify PNG magic header: 89 50 4E 47 0D 0A 1A 0A
            const isPng =
              imageBuffer[0] === 0x89 &&
              imageBuffer[1] === 0x50 &&
              imageBuffer[2] === 0x4e &&
              imageBuffer[3] === 0x47;

            computedImageSha256 = crypto.createHash('sha256').update(imageBuffer).digest('hex');
            imageVerified = isPng && imageBuffer.length === variation.fileSizeBytes;
          }
        }
      }

      // 8. Fetch Security Audit Logs for this case
      const auditLogs = await SecurityAudit.find({
        $or: [
          { 'details.caseId': caseId },
          { 'details.caseId': new Types.ObjectId(caseId) },
          { 'details.caseNumber': caseDoc.caseNumber },
        ],
      })
        .sort({ createdAt: -1 })
        .limit(40);

      // 9. Generate Report Number & Title
      const timestampSuffix = Date.now().toString().slice(-6);
      const randomSuffix = crypto.randomBytes(3).toString('hex').toUpperCase();
      const reportNumber = `RPT-${caseDoc.caseNumber.replace(/[^a-zA-Z0-9]/g, '')}-${timestampSuffix}-${randomSuffix}`;
      const title = input.title || `Forensic Facial Composite Dossier — Case ${caseDoc.caseNumber}`;

      // 10. Assemble PDF Data and render
      const pdfData: PDFReportData = {
        reportNumber,
        generatedAt: new Date(),
        generatedBy: userDoc,
        caseDoc,
        witnessDoc,
        compositeDoc,
        compositeHistory,
        auditLogs,
        imageVerified,
        computedImageSha256,
        notes: input.notes,
      };

      const pdfBuffer = await PDFReportService.generateForensicDossier(pdfData);

      // 11. Compute PDF cryptographic hash and save to disk
      const pdfSha256 = crypto.createHash('sha256').update(pdfBuffer).digest('hex');
      const filename = `report_${reportNumber}.pdf`;
      const filePath = path.resolve(REPORTS_STORAGE_DIR, filename);

      fs.writeFileSync(filePath, pdfBuffer);

      // 12. Persist Report metadata in MongoDB
      const reportDoc = await Report.create({
        reportNumber,
        case: caseDoc._id,
        caseNumber: caseDoc.caseNumber,
        witnessDescription: witnessDoc?._id,
        composite: compositeDoc?._id,
        compositeVersion: compositeDoc?.version,
        title,
        generatedBy: userDoc._id,
        investigatorName: userDoc.name,
        investigatorId: userDoc.investigatorId || userDoc.badgeNumber,
        storageKey: filename,
        fileSizeBytes: pdfBuffer.length,
        sha256: pdfSha256,
        status: 'generated',
        metadata: {
          hasWitnessStatement: Boolean(witnessDoc),
          hasComposite: Boolean(compositeDoc),
          hasAuditHistory: auditLogs.length > 0,
          imageVerified,
          imageSha256: computedImageSha256,
          checkpointSha256: compositeDoc?.generatorMetadata?.checkpointSha256,
        },
        notes: input.notes,
      });

      // 13. Audit successful generation
      await recordAudit(
        'REPORT_GENERATED_SUCCESS',
        'success',
        userId,
        {
          caseId,
          caseNumber: caseDoc.caseNumber,
          reportId: reportDoc._id.toString(),
          reportNumber,
          fileSizeBytes: pdfBuffer.length,
          pdfSha256,
          compositeVersion: compositeDoc?.version,
          imageVerified,
        },
        input.clientIp,
        input.userAgent
      );

      return reportDoc;
    } catch (err: any) {
      await recordAudit(
        'REPORT_GENERATION_FAILURE',
        'failure',
        userId,
        {
          caseId,
          error: err.message,
        },
        input.clientIp,
        input.userAgent
      );
      throw err;
    }
  }

  /**
   * Retrieves all reports for a specific case
   */
  public static async getReportsForCase(
    caseId: string,
    userId: string,
    userRole: string
  ): Promise<IReport[]> {
    await assertCaseAccess(caseId, userId, userRole);
    return Report.find({ case: caseId, status: { $ne: 'archived' } }).sort({ createdAt: -1 });
  }

  /**
   * Retrieves a single report metadata by ID
   */
  public static async getReportById(
    caseId: string,
    reportId: string,
    userId: string,
    userRole: string
  ): Promise<IReport> {
    await assertCaseAccess(caseId, userId, userRole);

    if (!Types.ObjectId.isValid(reportId)) {
      throw new ApiError('Invalid report ID format', 400);
    }

    const report = await Report.findOne({
      _id: reportId,
      case: caseId,
    });

    if (!report) {
      throw new ApiError('Forensic report not found for this case', 404);
    }

    return report;
  }

  /**
   * Retrieves the physical PDF file path for streaming download with strict path traversal protection
   */
  public static async getReportFile(
    caseId: string,
    reportId: string,
    userId: string,
    userRole: string,
    clientIp?: string,
    userAgent?: string
  ): Promise<{ filePath: string; filename: string; fileSizeBytes: number; mimeType: string }> {
    const report = await this.getReportById(caseId, reportId, userId, userRole);

    // Sanitize storage key against path traversal attacks
    const sanitizedKey = path.basename(report.storageKey);
    const resolvedPath = path.resolve(REPORTS_STORAGE_DIR, sanitizedKey);

    // Strict boundary enforcement: resolved path must reside strictly inside REPORTS_STORAGE_DIR
    if (!resolvedPath.startsWith(REPORTS_STORAGE_DIR)) {
      throw new ApiError('Illegal file path detected', 403);
    }

    if (!fs.existsSync(resolvedPath)) {
      throw new ApiError('Physical report dossier file is missing from archive storage', 404);
    }

    const stat = fs.statSync(resolvedPath);

    // Audit download
    await recordAudit(
      'REPORT_DOWNLOADED',
      'success',
      userId,
      {
        caseId,
        reportId: report._id.toString(),
        reportNumber: report.reportNumber,
        fileSizeBytes: stat.size,
      },
      clientIp,
      userAgent
    );

    return {
      filePath: resolvedPath,
      filename: `${report.reportNumber}.pdf`,
      fileSizeBytes: stat.size,
      mimeType: 'application/pdf',
    };
  }

  /**
   * Retrieves all reports across cases accessible to the authenticated user
   */
  public static async getAllReports(userId: string, userRole: string): Promise<IReport[]> {
    if (userRole === 'admin' || userRole === 'supervisor') {
      return Report.find({ status: { $ne: 'archived' } }).sort({ createdAt: -1 });
    }

    // For investigators, only return reports for cases they created or are assigned to
    const accessibleCases = await Case.find({
      isDeleted: false,
      $or: [{ createdBy: userId }, { assignedTo: userId }],
    }).select('_id');

    const caseIds = accessibleCases.map((c) => c._id);

    return Report.find({
      case: { $in: caseIds },
      status: { $ne: 'archived' },
    }).sort({ createdAt: -1 });
  }
}

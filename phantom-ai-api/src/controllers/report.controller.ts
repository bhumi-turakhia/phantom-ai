import { Request, Response } from 'express';
import fs from 'fs';
import { asyncHandler } from '../utils/asyncHandler';
import { ApiResponse } from '../utils/apiResponse';
import { ReportService } from '../services/report.service';

/**
 * @route   POST /api/v1/cases/:caseId/reports
 * @desc    Generates a complete forensic PDF dossier consolidating case, witness statement, NLP extractions, composite, lineage, and audit records
 * @access  Private (Authorized Investigator / Supervisor / Admin)
 */
export const generateReport = asyncHandler(async (req: Request, res: Response) => {
  const { caseId } = req.params;
  const user = (req as any).user;

  const result = await ReportService.generateReport(caseId, user.id, user.role, {
    witnessDescriptionId: req.body.witnessDescriptionId,
    compositeId: req.body.compositeId,
    compositeVersion:
      req.body.compositeVersion !== undefined ? Number(req.body.compositeVersion) : undefined,
    title: req.body.title,
    notes: req.body.notes,
    clientIp: req.ip || req.socket.remoteAddress,
    userAgent: req.get('user-agent'),
  });

  return ApiResponse.success(res, 'Forensic investigation report dossier generated successfully.', result, 201);
});

/**
 * @route   GET /api/v1/cases/:caseId/reports
 * @desc    Retrieves all generated report records for a given case
 * @access  Private
 */
export const getCaseReports = asyncHandler(async (req: Request, res: Response) => {
  const { caseId } = req.params;
  const user = (req as any).user;

  const reports = await ReportService.getReportsForCase(caseId, user.id, user.role);

  return ApiResponse.success(res, 'Case reports retrieved successfully.', reports);
});

/**
 * @route   GET /api/v1/cases/:caseId/reports/:reportId
 * @desc    Retrieves metadata for a specific forensic report
 * @access  Private
 */
export const getReportById = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, reportId } = req.params;
  const user = (req as any).user;

  const report = await ReportService.getReportById(caseId, reportId, user.id, user.role);

  return ApiResponse.success(res, 'Report details retrieved successfully.', report);
});

/**
 * @route   GET /api/v1/cases/:caseId/reports/:reportId/download
 * @desc    Downloads the physical forensic investigation dossier PDF with path traversal protection
 * @access  Private
 */
export const downloadReportFile = asyncHandler(async (req: Request, res: Response) => {
  const { caseId, reportId } = req.params;
  const user = (req as any).user;
  const clientIp = req.ip || req.socket.remoteAddress;
  const userAgent = req.get('user-agent');

  const { filePath, filename, fileSizeBytes, mimeType } = await ReportService.getReportFile(
    caseId,
    reportId,
    user.id,
    user.role,
    clientIp,
    userAgent
  );

  res.setHeader('Content-Type', mimeType);
  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
  res.setHeader('Content-Length', fileSizeBytes);

  const fileStream = fs.createReadStream(filePath);
  fileStream.pipe(res);
});

/**
 * @route   GET /api/v1/reports
 * @desc    Retrieves all accessible reports across cases for the authenticated user
 * @access  Private
 */
export const getAllReports = asyncHandler(async (req: Request, res: Response) => {
  const user = (req as any).user;

  const reports = await ReportService.getAllReports(user.id, user.role);

  return ApiResponse.success(res, 'Accessible forensic reports retrieved successfully.', reports);
});


import api from './api';

export interface ReportMetadata {
  hasWitnessStatement: boolean;
  hasComposite: boolean;
  hasAuditHistory: boolean;
  imageVerified: boolean;
  imageSha256?: string;
  checkpointSha256?: string;
  pageCount?: number;
}

export interface ReportItem {
  _id: string;
  reportNumber: string;
  case: string;
  caseNumber: string;
  witnessDescription?: string;
  composite?: string;
  compositeVersion?: number;
  title: string;
  generatedBy: string;
  investigatorName: string;
  investigatorId?: string;
  storageKey: string;
  fileSizeBytes: number;
  sha256: string;
  status: 'generated' | 'archived';
  metadata: ReportMetadata;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

export interface GenerateReportPayload {
  witnessDescriptionId?: string;
  compositeId?: string;
  compositeVersion?: number;
  title?: string;
  notes?: string;
}

/**
 * Generates a new multi-modal forensic PDF dossier for a case
 */
export const generateReport = async (
  caseId: string,
  payload: GenerateReportPayload = {}
): Promise<ReportItem> => {
  const response = await api.post(`/cases/${caseId}/reports`, payload);
  return response.data.data;
};

/**
 * Retrieves all forensic reports for a specific case
 */
export const getCaseReports = async (caseId: string): Promise<ReportItem[]> => {
  const response = await api.get(`/cases/${caseId}/reports`);
  return response.data.data;
};

/**
 * Retrieves all accessible forensic reports across cases for the investigator
 */
export const getAllReports = async (): Promise<ReportItem[]> => {
  const response = await api.get('/reports');
  return response.data.data;
};

/**
 * Retrieves a single report metadata
 */
export const getReportById = async (caseId: string, reportId: string): Promise<ReportItem> => {
  const response = await api.get(`/cases/${caseId}/reports/${reportId}`);
  return response.data.data;
};

/**
 * Downloads the physical PDF dossier file directly to the client browser
 */
export const downloadReportPdf = async (
  caseId: string,
  reportId: string,
  filename: string
): Promise<void> => {
  const response = await api.get(`/cases/${caseId}/reports/${reportId}/download`, {
    responseType: 'blob',
  });

  const blob = new Blob([response.data], { type: 'application/pdf' });
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.setAttribute('download', filename.endsWith('.pdf') ? filename : `${filename}.pdf`);
  document.body.appendChild(link);
  link.click();
  link.parentNode?.removeChild(link);
  window.URL.revokeObjectURL(downloadUrl);
};


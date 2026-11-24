import { useState, useEffect } from 'react';
import {
  FileBarChart,
  Download,
  FileText,
  ShieldCheck,
  Clock,
  Layers,
  Sparkles,
  AlertTriangle,
  RefreshCw,
  HardDrive,
  Hash,
  CheckCircle2,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { EvidenceTag } from '../components/ui/EvidenceTag';
import { StatusBadge } from '../components/ui/Badge';
import { PageTransition } from '../components/layout/PageTransition';
import { EmptyState } from '../components/ui/EmptyState';
import { getAllCases } from '../services/caseService';
import { getWitnessDescriptionsByCase, getCaseComposites } from '../services/imageService';
import {
  generateReport,
  getCaseReports,
  getAllReports,
  downloadReportPdf,
} from '../services/reportService';
import type { ReportItem } from '../services/reportService';

export default function Reports() {
  const [cases, setCases] = useState<any[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [witnessDescriptions, setWitnessDescriptions] = useState<any[]>([]);
  const [selectedWitnessId, setSelectedWitnessId] = useState<string>('');
  const [composites, setComposites] = useState<any[]>([]);
  const [selectedCompositeId, setSelectedCompositeId] = useState<string>('');

  const [reports, setReports] = useState<ReportItem[]>([]);
  const [reportTitle, setReportTitle] = useState<string>('');
  const [reportNotes, setReportNotes] = useState<string>('');

  const [isLoadingCases, setIsLoadingCases] = useState<boolean>(true);
  const [isLoadingDetails, setIsLoadingDetails] = useState<boolean>(false);
  const [isLoadingReports, setIsLoadingReports] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Load initial cases and reports
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setIsLoadingCases(true);
    setErrorMsg(null);
    try {
      const casesRes = await getAllCases();
      const rawCases = casesRes?.data?.cases || casesRes?.cases || casesRes?.data || [];
      const casesList = Array.isArray(rawCases) ? rawCases : [];
      setCases(casesList);

      if (casesList.length > 0) {
        setSelectedCaseId(casesList[0]._id);
      }

      await loadReports(casesList.length > 0 ? casesList[0]._id : undefined);
    } catch (err: any) {
      console.error('Failed to load initial cases:', err);
      setErrorMsg('Failed to fetch case files. Ensure backend API is operational.');
    } finally {
      setIsLoadingCases(false);
    }
  };

  // When selected case changes, fetch witness descriptions, composites, and case reports
  useEffect(() => {
    if (!selectedCaseId) {
      setWitnessDescriptions([]);
      setComposites([]);
      setSelectedWitnessId('');
      setSelectedCompositeId('');
      return;
    }

    loadCaseDetails(selectedCaseId);
    loadReports(selectedCaseId);
  }, [selectedCaseId]);

  const loadCaseDetails = async (caseId: string) => {
    setIsLoadingDetails(true);
    try {
      const [witnessRes, compositeRes] = await Promise.all([
        getWitnessDescriptionsByCase(caseId).catch(() => ({ data: [] })),
        getCaseComposites(caseId).catch(() => ({ data: [] })),
      ]);

      const rawWitness =
        witnessRes?.data?.witnessDescriptions ||
        witnessRes?.witnessDescriptions ||
        witnessRes?.data ||
        witnessRes ||
        [];
      const wList = Array.isArray(rawWitness) ? rawWitness : [];

      const rawComposite =
        compositeRes?.data?.composites ||
        compositeRes?.composites ||
        compositeRes?.data ||
        compositeRes ||
        [];
      const cList = Array.isArray(rawComposite) ? rawComposite : [];

      setWitnessDescriptions(wList);
      setComposites(cList);

      if (wList.length > 0) {
        setSelectedWitnessId(wList[0]._id);
      } else {
        setSelectedWitnessId('');
      }

      if (cList.length > 0) {
        // Prefer approved composite, else newest
        const approved = cList.find((c: any) => c.status === 'approved') || cList[0];
        setSelectedCompositeId(approved._id);
      } else {
        setSelectedCompositeId('');
      }
    } catch (err) {
      console.error('Error fetching case details:', err);
    } finally {
      setIsLoadingDetails(false);
    }
  };

  const loadReports = async (caseId?: string) => {
    setIsLoadingReports(true);
    try {
      let data: any = [];
      if (caseId) {
        data = await getCaseReports(caseId);
      } else {
        data = await getAllReports();
      }
      const rawReports = data?.data?.reports || data?.reports || data?.data || data || [];
      const rList = Array.isArray(rawReports) ? rawReports : [];
      setReports(rList);
    } catch (err: any) {
      console.error('Error fetching reports:', err);
      setErrorMsg('Failed to load generated reports.');
    } finally {
      setIsLoadingReports(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!selectedCaseId) {
      setErrorMsg('Please select a valid case before generating a forensic dossier.');
      return;
    }

    setIsGenerating(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const newReport = await generateReport(selectedCaseId, {
        witnessDescriptionId: selectedWitnessId || undefined,
        compositeId: selectedCompositeId || undefined,
        title: reportTitle.trim() || undefined,
        notes: reportNotes.trim() || undefined,
      });

      setSuccessMsg(`Forensic dossier ${newReport.reportNumber} generated and verified.`);
      setReportTitle('');
      setReportNotes('');
      await loadReports(selectedCaseId);
    } catch (err: any) {
      console.error('Report generation error:', err);
      setErrorMsg(
        err.response?.data?.message || err.message || 'Failed to generate forensic PDF report.'
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async (report: ReportItem) => {
    setDownloadingId(report._id);
    setErrorMsg(null);
    try {
      await downloadReportPdf(report.case, report._id, `${report.reportNumber}.pdf`);
    } catch (err: any) {
      console.error('Download error:', err);
      setErrorMsg('Failed to stream PDF report from secure archive.');
    } finally {
      setDownloadingId(null);
    }
  };

  const selectedCase = cases.find((c) => c._id === selectedCaseId);
  const selectedComposite = composites.find((c) => c._id === selectedCompositeId);
  const selectedWitness = witnessDescriptions.find((w) => w._id === selectedWitnessId);

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '0 B';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <PageTransition>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-ink-primary flex items-center gap-2">
              <FileBarChart className="h-5 w-5 text-gold" />
              Forensic Investigation Dossier & Evidence Packaging
            </h1>
            <p className="label-technical text-ink-disabled mt-1">
              Multi-modal forensic investigation dossier synthesis · Lossless composite embedding · Cryptographic provenance audit
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                loadInitialData();
              }}
              disabled={isLoadingCases || isLoadingReports}
            >
              <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${isLoadingReports ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>

        {/* Notifications */}
        {errorMsg && (
          <div className="rounded border border-status-critical/40 bg-status-critical/10 px-4 py-3 text-sm text-status-critical flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-xs text-status-critical hover:underline ml-4"
            >
              Dismiss
            </button>
          </div>
        )}

        {successMsg && (
          <div className="rounded border border-status-verified/40 bg-status-verified/10 px-4 py-3 text-sm text-status-verified flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>{successMsg}</span>
            </div>
            <button
              onClick={() => setSuccessMsg(null)}
              className="text-xs text-status-verified hover:underline ml-4"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Section 1: Configuration & Generation Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Generation Parameters */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FileText className="h-4 w-4 text-gold" />
                Generate New Forensic Dossier
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Select Case */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-ink-secondary flex items-center justify-between">
                  <span>Target Investigation Case</span>
                  {selectedCase && (
                    <span className="text-xs text-ink-disabled font-mono">
                      ID: {selectedCase._id.slice(-8)}
                    </span>
                  )}
                </label>
                <select
                  value={selectedCaseId}
                  onChange={(e) => setSelectedCaseId(e.target.value)}
                  disabled={isLoadingCases || isGenerating}
                  className="w-full rounded border border-line bg-bg-inset px-3 py-2 text-sm text-ink-primary focus:border-gold focus:outline-none"
                >
                  {cases.length === 0 ? (
                    <option value="">No active cases found</option>
                  ) : (
                    cases.map((c) => (
                      <option key={c._id} value={c._id}>
                        {c.caseNumber} — {c.title} ({c.status || 'active'})
                      </option>
                    ))
                  )}
                </select>
              </div>

              {/* Witness & Composite Selectors */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                {/* Witness Statement */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-ink-secondary flex items-center justify-between">
                    <span>Witness Statement</span>
                    <span className="text-[11px] text-ink-disabled">
                      {witnessDescriptions.length} available
                    </span>
                  </label>
                  <select
                    value={selectedWitnessId}
                    onChange={(e) => setSelectedWitnessId(e.target.value)}
                    disabled={isLoadingDetails || witnessDescriptions.length === 0 || isGenerating}
                    className="w-full rounded border border-line bg-bg-inset px-3 py-2 text-xs text-ink-primary focus:border-gold focus:outline-none"
                  >
                    {witnessDescriptions.length === 0 ? (
                      <option value="">No witness statements recorded</option>
                    ) : (
                      witnessDescriptions.map((w, idx) => (
                        <option key={w._id} value={w._id}>
                          Statement #{idx + 1} ({w.witnessName || 'Anonymous'} -{' '}
                          {formatDate(w.createdAt)})
                        </option>
                      ))
                    )}
                  </select>
                </div>

                {/* Facial Composite Version */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-ink-secondary flex items-center justify-between">
                    <span>Facial Composite Version</span>
                    <span className="text-[11px] text-ink-disabled">
                      {composites.length} versions
                    </span>
                  </label>
                  <select
                    value={selectedCompositeId}
                    onChange={(e) => setSelectedCompositeId(e.target.value)}
                    disabled={isLoadingDetails || composites.length === 0 || isGenerating}
                    className="w-full rounded border border-line bg-bg-inset px-3 py-2 text-xs text-ink-primary focus:border-gold focus:outline-none"
                  >
                    {composites.length === 0 ? (
                      <option value="">No facial composites generated yet</option>
                    ) : (
                      composites.map((comp) => (
                        <option key={comp._id} value={comp._id}>
                          v{comp.version} — {comp.status} (Seed:{' '}
                          {comp.variations?.[0]?.seed ?? 'N/A'},{' '}
                          {comp.editingType || 'initial'})
                        </option>
                      ))
                    )}
                  </select>
                </div>
              </div>

              {/* Title & Custom Notes */}
              <div className="space-y-1.5 pt-2">
                <label className="text-xs font-medium text-ink-secondary">
                  Custom Dossier Title (Optional)
                </label>
                <input
                  type="text"
                  placeholder={
                    selectedCase
                      ? `Forensic Facial Composite Dossier — Case ${selectedCase.caseNumber}`
                      : 'Forensic Facial Composite Dossier'
                  }
                  value={reportTitle}
                  onChange={(e) => setReportTitle(e.target.value)}
                  disabled={isGenerating}
                  className="w-full rounded border border-line bg-bg-inset px-3 py-2 text-xs text-ink-primary focus:border-gold focus:outline-none"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-ink-secondary">
                  Investigator Addendum / Evidentiary Notes (Optional)
                </label>
                <textarea
                  rows={2}
                  placeholder="Record additional forensic context, witness confidence remarks, or investigative handling instructions..."
                  value={reportNotes}
                  onChange={(e) => setReportNotes(e.target.value)}
                  disabled={isGenerating}
                  className="w-full rounded border border-line bg-bg-inset px-3 py-2 text-xs text-ink-primary focus:border-gold focus:outline-none"
                />
              </div>

              {/* Action Button */}
              <div className="pt-2 flex items-center justify-between border-t border-line">
                <p className="text-[11px] text-ink-disabled">
                  PDF document is cryptographically hashed with SHA-256 upon assembly.
                </p>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleGenerateReport}
                  disabled={isGenerating || !selectedCaseId}
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                      Rendering Dossier...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                      Synthesize Forensic PDF
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Right Col: Dossier Specification & Live Inclusion Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Layers className="h-4 w-4 text-ink-tertiary" />
                Dossier Sections Overview
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2.5 text-xs">
              <div className="rounded border border-line bg-bg-inset p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink-primary">1. Official Case Header</span>
                  <StatusBadge tone={selectedCase ? 'verified' : 'pending'}>
                    {selectedCase ? 'Ready' : 'Pending'}
                  </StatusBadge>
                </div>
                <p className="text-[11px] text-ink-disabled">
                  Agency badge, case ID, incident date, assigned investigator.
                </p>
              </div>

              <div className="rounded border border-line bg-bg-inset p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink-primary">2. Verbatim Statement</span>
                  <StatusBadge tone={selectedWitness ? 'verified' : 'pending'}>
                    {selectedWitness ? 'Included' : 'Empty'}
                  </StatusBadge>
                </div>
                <p className="text-[11px] text-ink-disabled">
                  Immutable witness testimony preserved verbatim.
                </p>
              </div>

              <div className="rounded border border-line bg-bg-inset p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink-primary">3. NLP Extractions & Audit</span>
                  <StatusBadge tone={selectedWitness ? 'verified' : 'pending'}>
                    {selectedWitness ? 'Structured' : 'None'}
                  </StatusBadge>
                </div>
                <p className="text-[11px] text-ink-disabled">
                  BiLSTM-CRF spans, polarity badges, investigator modifications.
                </p>
              </div>

              <div className="rounded border border-line bg-bg-inset p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink-primary">4. Facial Composite Image</span>
                  <StatusBadge tone={selectedComposite ? 'verified' : 'pending'}>
                    {selectedComposite ? `v${selectedComposite.version}` : 'Not run'}
                  </StatusBadge>
                </div>
                <p className="text-[11px] text-ink-disabled">
                  Lossless embedded PNG with 20 GCIT condition labels.
                </p>
              </div>

              <div className="rounded border border-line bg-bg-inset p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink-primary">5-8. Lineage, Hash & Rules</span>
                  <StatusBadge tone="active">Enforced</StatusBadge>
                </div>
                <p className="text-[11px] text-ink-disabled">
                  Chain-of-custody, SHA-256 checks, 8-clause disclaimer.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Section 2: Archive of Generated Reports */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-ink-tertiary" />
                Case Investigation Dossiers Archive
              </CardTitle>
              <p className="text-xs text-ink-disabled mt-0.5">
                {selectedCase
                  ? `Showing generated PDF reports for Case ${selectedCase.caseNumber}`
                  : 'All agency forensic reports'}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="label-technical text-xs text-ink-secondary">
                {reports.length} {reports.length === 1 ? 'dossier' : 'dossiers'} on file
              </span>
            </div>
          </CardHeader>

          <CardContent>
            {isLoadingReports ? (
              <div className="py-12 flex flex-col items-center justify-center gap-2 text-ink-secondary">
                <RefreshCw className="h-5 w-5 animate-spin text-gold" />
                <span className="text-xs">Loading report archive records...</span>
              </div>
            ) : reports.length === 0 ? (
              <EmptyState
                title="No forensic reports generated"
                description={
                  selectedCase
                    ? `No PDF dossiers have been compiled for Case ${selectedCase.caseNumber} yet. Click "Synthesize Forensic PDF" above to compile the forensic investigation dossier.`
                    : 'Select a case above to view or synthesize a forensic dossier.'
                }
                icon={<FileBarChart className="h-10 w-10 text-ink-disabled" />}
              />
            ) : (
              <div className="space-y-3">
                {reports.map((report) => (
                  <div
                    key={report._id}
                    className="flex flex-col sm:flex-row sm:items-center justify-between rounded border border-line bg-bg-inset p-4 gap-4 hover:border-line-hover transition-colors"
                  >
                    {/* Left: Metadata */}
                    <div className="space-y-1.5 flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <EvidenceTag caseId={report.caseNumber} />
                        <span className="text-xs font-mono font-semibold text-ink-primary">
                          {report.reportNumber}
                        </span>
                        {report.compositeVersion && (
                          <span className="rounded bg-navy px-2 py-0.5 text-[10px] font-mono text-gold border border-gold/30">
                            v{report.compositeVersion}
                          </span>
                        )}
                        <StatusBadge tone="verified">Verified</StatusBadge>
                        {report.metadata?.imageVerified && (
                          <span className="inline-flex items-center gap-1 text-[11px] text-status-verified font-mono">
                            <ShieldCheck className="h-3 w-3" />
                            Image Hash Match
                          </span>
                        )}
                      </div>

                      <h4 className="text-sm font-medium text-ink-primary truncate">
                        {report.title}
                      </h4>

                      <div className="flex items-center gap-4 text-xs text-ink-disabled flex-wrap">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatDate(report.createdAt)}
                        </span>
                        <span>Investigator: {report.investigatorName}</span>
                        <span>Size: {formatFileSize(report.fileSizeBytes)}</span>
                      </div>

                      {/* Cryptographic SHA-256 Preview */}
                      <div className="flex items-center gap-1.5 pt-1 text-[11px] font-mono text-ink-tertiary">
                        <Hash className="h-3 w-3 shrink-0 text-ink-disabled" />
                        <span className="text-ink-disabled">SHA-256:</span>
                        <span className="truncate max-w-xs">{report.sha256}</span>
                      </div>
                    </div>

                    {/* Right: Actions */}
                    <div className="flex items-center gap-3 shrink-0">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleDownload(report)}
                        disabled={downloadingId === report._id}
                        className="flex items-center gap-1.5"
                      >
                        {downloadingId === report._id ? (
                          <>
                            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                            <span>Streaming...</span>
                          </>
                        ) : (
                          <>
                            <Download className="h-3.5 w-3.5" />
                            <span>Download PDF</span>
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageTransition>
  );
}

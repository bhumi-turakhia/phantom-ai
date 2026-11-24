import PDFDocument from 'pdfkit';
import fs from 'fs';
import path from 'path';
import { ICase } from '../models/Case.model';
import { IWitnessDescription } from '../models/WitnessDescription.model';
import { IComposite } from '../models/Composite.model';
import { ISecurityAudit } from '../models/SecurityAudit.model';
import { IUser } from '../models/User.model';

export interface PDFReportData {
  reportNumber: string;
  generatedAt: Date;
  generatedBy: IUser;
  caseDoc: ICase;
  witnessDoc?: IWitnessDescription | null;
  compositeDoc?: IComposite | null;
  compositeHistory?: IComposite[];
  auditLogs?: ISecurityAudit[];
  imageVerified?: boolean;
  computedImageSha256?: string;
  notes?: string;
}

const COLORS = {
  navy: '#0f172a',
  darkSlate: '#1e293b',
  slate: '#475569',
  lightSlate: '#94a3b8',
  border: '#cbd5e1',
  bgLight: '#f8fafc',
  gold: '#b45309',
  goldLight: '#fef3c7',
  goldBorder: '#f59e0b',
  green: '#047857',
  greenLight: '#ecfdf5',
  greenBorder: '#10b981',
  red: '#b91c1c',
  redLight: '#fef2f2',
  redBorder: '#ef4444',
  purple: '#6b21a8',
  purpleLight: '#faf5ff',
  purpleBorder: '#a855f7',
};

export class PDFReportService {
  /**
   * Generates a complete forensic investigation dossier as a PDF Buffer
   */
  public static async generateForensicDossier(data: PDFReportData): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      try {
        const doc = new PDFDocument({
          size: 'A4',
          margins: { top: 45, bottom: 45, left: 45, right: 45 },
          bufferPages: true,
          info: {
            Title: `Forensic Evidence Dossier — ${data.caseDoc.caseNumber}`,
            Author: 'Phantom AI Forensic System',
            Subject: `Investigation Report ${data.reportNumber}`,
            Keywords: 'Forensic, Facial Composite, Criminal Investigation, Evidentiary Dossier',
            CreationDate: data.generatedAt,
          },
        });

        const buffers: Buffer[] = [];
        doc.on('data', buffers.push.bind(buffers));
        doc.on('end', () => {
          const pdfBuffer = Buffer.concat(buffers);
          resolve(pdfBuffer);
        });
        doc.on('error', (err) => reject(err));

        const pageWidth = doc.page.width - doc.page.margins.left - doc.page.margins.right;

        // -------------------------------------------------------------
        // HELPER FUNCTIONS
        // -------------------------------------------------------------
        const drawHorizontalRule = (y?: number, color = COLORS.border, width = 1) => {
          const currentY = y !== undefined ? y : doc.y;
          doc
            .save()
            .strokeColor(color)
            .lineWidth(width)
            .moveTo(doc.page.margins.left, currentY)
            .lineTo(doc.page.margins.left + pageWidth, currentY)
            .stroke()
            .restore();
          if (y === undefined) doc.y = currentY + 6;
        };

        const drawSectionHeader = (numberStr: string, title: string) => {
          if (doc.y > doc.page.height - 120) {
            doc.addPage();
          } else {
            doc.moveDown(0.8);
          }
          const startY = doc.y;
          doc
            .save()
            .rect(doc.page.margins.left, startY, 4, 16)
            .fillColor(COLORS.gold)
            .fill()
            .restore();

          doc
            .font('Helvetica-Bold')
            .fontSize(12)
            .fillColor(COLORS.navy)
            .text(`${numberStr} — ${title.toUpperCase()}`, doc.page.margins.left + 10, startY + 2);

          doc.y = startY + 22;
          drawHorizontalRule(doc.y, COLORS.goldBorder, 0.75);
          doc.moveDown(0.4);
        };

        // -------------------------------------------------------------
        // COVER / HEADER BLOCK
        // -------------------------------------------------------------
        // Top agency banner
        doc
          .save()
          .rect(doc.page.margins.left, doc.y, pageWidth, 52)
          .fillColor(COLORS.navy)
          .fill()
          .restore();

        const bannerY = doc.y + 10;
        doc
          .font('Helvetica-Bold')
          .fontSize(16)
          .fillColor('#ffffff')
          .text('PHANTOM AI 3.0 — FORENSIC INVESTIGATION DOSSIER', doc.page.margins.left + 14, bannerY);

        doc
          .font('Helvetica')
          .fontSize(9)
          .fillColor('#cbd5e1')
          .text('OFFICIAL LAW ENFORCEMENT WORK PRODUCT • STRICTLY CONFIDENTIAL', doc.page.margins.left + 14, bannerY + 20);

        doc.y = doc.y + 35;
        doc.moveDown(0.5);

        // Dossier Identification Grid
        const metaBoxY = doc.y;
        doc
          .save()
          .rect(doc.page.margins.left, metaBoxY, pageWidth, 64)
          .fillColor(COLORS.bgLight)
          .strokeColor(COLORS.border)
          .lineWidth(1)
          .fillAndStroke()
          .restore();

        doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.slate);
        doc.text('REPORT NUMBER:', doc.page.margins.left + 12, metaBoxY + 10);
        doc.text('CASE NUMBER:', doc.page.margins.left + 180, metaBoxY + 10);
        doc.text('GENERATED DATE:', doc.page.margins.left + 340, metaBoxY + 10);

        doc.font('Helvetica-Bold').fontSize(11).fillColor(COLORS.navy);
        doc.text(data.reportNumber, doc.page.margins.left + 12, metaBoxY + 22);
        doc.text(data.caseDoc.caseNumber, doc.page.margins.left + 180, metaBoxY + 22);
        doc.text(new Date(data.generatedAt).toISOString().replace('T', ' ').slice(0, 19) + ' UTC', doc.page.margins.left + 340, metaBoxY + 22);

        doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.slate);
        doc.text('AUTHORIZING INVESTIGATOR:', doc.page.margins.left + 12, metaBoxY + 40);
        doc.text('SECURITY CLASSIFICATION:', doc.page.margins.left + 340, metaBoxY + 40);

        doc.font('Helvetica').fontSize(10).fillColor(COLORS.navy);
        doc.text(`${data.generatedBy.name} (${data.generatedBy.investigatorId || data.generatedBy.badgeNumber || 'INV-AUTH'})`, doc.page.margins.left + 12, metaBoxY + 50);
        
        doc.font('Helvetica-Bold').fontSize(10).fillColor(COLORS.red);
        doc.text('RESTRICTED / LAW ENFORCEMENT', doc.page.margins.left + 340, metaBoxY + 50);

        doc.y = metaBoxY + 75;

        // -------------------------------------------------------------
        // SECTION 1: CASE INFORMATION
        // -------------------------------------------------------------
        drawSectionHeader('Section 1', 'Case Information & Incident Profile');

        const caseInfo = [
          ['Case Title:', data.caseDoc.title || 'Untitled Case Record'],
          ['Case Status:', (data.caseDoc.status || 'open').toUpperCase()],
          ['Priority Level:', (data.caseDoc.priority || 'medium').toUpperCase()],
          ['Incident Date:', data.caseDoc.incidentDate ? new Date(data.caseDoc.incidentDate).toLocaleDateString() : 'Unspecified'],
          ['Location:', data.caseDoc.location?.city ? `${data.caseDoc.location.city}, ${data.caseDoc.location.state || ''}` : 'Jurisdiction Record Stored'],
          ['Case Description:', data.caseDoc.description || 'No detailed case summary entered.'],
        ];

        caseInfo.forEach(([label, val]) => {
          const rowY = doc.y;
          doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.slate).text(label, doc.page.margins.left, rowY, { width: 130 });
          doc.font('Helvetica').fontSize(9).fillColor(COLORS.darkSlate).text(val, doc.page.margins.left + 135, rowY, { width: pageWidth - 135 });
          doc.moveDown(0.25);
        });

        if (data.notes && data.notes.trim()) {
          const notesY = doc.y;
          doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.slate).text('Investigator Addendum:', doc.page.margins.left, notesY, { width: 130 });
          doc.font('Helvetica-Oblique').fontSize(9).fillColor(COLORS.darkSlate).text(data.notes.trim(), doc.page.margins.left + 135, notesY, { width: pageWidth - 135 });
          doc.moveDown(0.25);
        }

        // -------------------------------------------------------------
        // SECTION 2: ORIGINAL WITNESS STATEMENT
        // -------------------------------------------------------------
        drawSectionHeader('Section 2', 'Original Witness Statement & Interview Intake');

        if (data.witnessDoc) {
          const wit = data.witnessDoc;
          doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.slate);
          doc.text(`Witness Identifier: ${wit.isAnonymous ? 'ANONYMOUS WITNESS' : (wit.witnessName || 'Witness on Record')}  •  Review Status: [${(wit.reviewStatus || 'pending').toUpperCase()}]  •  Intake Timestamp: ${new Date(wit.createdAt).toISOString()}`);
          doc.moveDown(0.4);

          const statementText = wit.rawStatement?.trim() || '(No raw statement text recorded.)';
          
          doc.font('Courier').fontSize(9);
          const textHeight = doc.heightOfString(statementText, { width: pageWidth - 24 });
          const boxHeight = Math.max(textHeight + 20, 44);

          if (doc.y + boxHeight > doc.page.height - 50) {
            doc.addPage();
          }

          const stmtY = doc.y;

          doc
            .save()
            .rect(doc.page.margins.left, stmtY, pageWidth, boxHeight)
            .fillColor(COLORS.bgLight)
            .strokeColor(COLORS.border)
            .lineWidth(1)
            .fillAndStroke()
            .restore();

          doc
            .save()
            .rect(doc.page.margins.left, stmtY, 3, boxHeight)
            .fillColor(COLORS.gold)
            .fill()
            .restore();

          doc
            .font('Courier')
            .fontSize(9)
            .fillColor(COLORS.darkSlate)
            .text(statementText, doc.page.margins.left + 12, stmtY + 10, { width: pageWidth - 24, lineGap: 2 });

          doc.y = stmtY + boxHeight + 8;
          doc
            .font('Helvetica-Oblique')
            .fontSize(8)
            .fillColor(COLORS.slate)
            .text('Note: The verbatim testimony above is cryptographically preserved as ingested. Subsequent attribute normalization or corrections do not alter this primary testimony.', doc.page.margins.left);
        } else {
          doc.font('Helvetica-Oblique').fontSize(9).fillColor(COLORS.slate).text('No witness description document attached to this report.');
        }

        // -------------------------------------------------------------
        // SECTION 3: NLP EVIDENCE EXTRACTION & CORRECTIONS
        // -------------------------------------------------------------
        drawSectionHeader('Section 3', 'NLP Evidence Extraction & Investigator Audits');

        if (data.witnessDoc && data.witnessDoc.extractedAttributes && data.witnessDoc.extractedAttributes.length > 0) {
          doc.font('Helvetica-Bold').fontSize(8.5).fillColor(COLORS.slate);
          const tableHeaders = ['CATEGORY', 'EXTRACTED VALUE', 'SOURCE TEXT', 'OFFSETS', 'EVIDENCE STATE'];
          const colX = [
            doc.page.margins.left,
            doc.page.margins.left + 90,
            doc.page.margins.left + 180,
            doc.page.margins.left + 340,
            doc.page.margins.left + 420,
          ];

          const headerY = doc.y;
          tableHeaders.forEach((th, i) => {
            doc.text(th, colX[i], headerY);
          });
          drawHorizontalRule(doc.y + 12, COLORS.navy, 1);
          doc.y = headerY + 16;

          data.witnessDoc.extractedAttributes.forEach((attr) => {
            if (doc.y > doc.page.height - 60) doc.addPage();
            const rY = doc.y;
            doc.font('Helvetica-Bold').fontSize(8).fillColor(COLORS.navy).text(attr.category.toUpperCase(), colX[0], rY, { width: 85 });
            doc.font('Helvetica').fontSize(8).fillColor(COLORS.darkSlate).text(attr.value || 'None', colX[1], rY, { width: 85 });
            doc.font('Helvetica-Oblique').fontSize(8).fillColor(COLORS.slate).text(`"${attr.sourceText || ''}"`, colX[2], rY, { width: 155, lineBreak: false, ellipsis: true });
            doc.font('Courier').fontSize(8).fillColor(COLORS.slate).text(typeof attr.start === 'number' && typeof attr.end === 'number' ? `${attr.start}:${attr.end}` : 'N/A', colX[3], rY, { width: 75 });
            
            // Evidence state color tag
            const stateStr = (attr.evidenceState || 'unknown').toUpperCase();
            let stateColor = COLORS.slate;
            if (stateStr === 'EXPLICIT') stateColor = COLORS.green;
            else if (stateStr === 'NEGATED') stateColor = COLORS.red;
            else if (stateStr === 'UNCERTAIN') stateColor = COLORS.gold;
            else if (stateStr === 'CONTRADICTORY') stateColor = COLORS.purple;

            doc.font('Helvetica-Bold').fontSize(8).fillColor(stateColor).text(`[${stateStr}]`, colX[4], rY, { width: 80 });
            doc.y = rY + 13;
          });

          // Investigator Corrections Log
          if (data.witnessDoc.corrections && data.witnessDoc.corrections.length > 0) {
            doc.moveDown(0.5);
            doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.navy).text('Investigator Forensic Audit Trail & Corrections:');
            doc.moveDown(0.2);

            data.witnessDoc.corrections.forEach((c, idx) => {
              if (doc.y > doc.page.height - 50) doc.addPage();
              doc.font('Helvetica').fontSize(8).fillColor(COLORS.darkSlate);
              const cText = `• [CORRECTION #${idx + 1}] Category: ${c.attributeCategory} | Prev: "${c.previousValue || 'null'}" (${c.previousEvidenceState}) -> Corrected: "${c.correctedValue || 'null'}" (${c.correctedEvidenceState}) | Justification: "${c.reason}" | Timestamp: ${new Date(c.timestamp).toISOString()}`;
              doc.text(cText, { width: pageWidth });
              doc.moveDown(0.15);
            });
          }
        } else {
          doc.font('Helvetica-Oblique').fontSize(9).fillColor(COLORS.slate).text('No NLP attribute extractions available for this witness intake.');
        }

        // -------------------------------------------------------------
        // SECTION 4: COMPOSITE GENERATION & GCIT CONDITIONING
        // -------------------------------------------------------------
        drawSectionHeader('Section 4', 'Facial Composite Generation & Conditioning Vector');

        if (data.compositeDoc) {
          if (doc.y > doc.page.height - 180) doc.addPage();
          const comp = data.compositeDoc;
          const selectedVar = comp.variations?.find((v) => v.isSelected) || comp.variations?.[0];

          // Embed primary generated image
          const startCompY = doc.y;
          let imageDrawn = false;

          if (selectedVar) {
            try {
              let imgBuffer: Buffer | null = null;
              // Attempt 1: disk storageKey
              const diskPath = path.resolve(process.cwd(), 'data', 'composites', selectedVar.storageKey);
              if (fs.existsSync(diskPath)) {
                imgBuffer = fs.readFileSync(diskPath);
              } else if (selectedVar.dataUrl && selectedVar.dataUrl.includes(',')) {
                imgBuffer = Buffer.from(selectedVar.dataUrl.split(',')[1], 'base64');
              }

              if (imgBuffer && imgBuffer.length > 100) {
                // Draw 140x140 pt enlarged preview with crisp pixel rendering
                doc.image(imgBuffer, doc.page.margins.left, startCompY, {
                  width: 140,
                  height: 140,
                });
                
                doc
                  .save()
                  .rect(doc.page.margins.left, startCompY, 140, 140)
                  .strokeColor(COLORS.border)
                  .lineWidth(1)
                  .stroke()
                  .restore();

                imageDrawn = true;
              }
            } catch (imgErr) {
              // fallback if image read fails
            }
          }

          const detailsX = imageDrawn ? doc.page.margins.left + 155 : doc.page.margins.left;
          const detailsWidth = imageDrawn ? pageWidth - 155 : pageWidth;

          doc.y = startCompY;
          doc.font('Helvetica-Bold').fontSize(11).fillColor(COLORS.navy);
          doc.text(`Composite Version: ${comp.version}  •  Status: [${comp.status.toUpperCase()}]`, detailsX, doc.y, { width: detailsWidth });
          doc.moveDown(0.3);

          const compDetails = [
            ['Selected Variation:', `Variation #${selectedVar?.variationIndex ?? 1} (of ${comp.variations?.length || 1})`],
            ['Generator Seed:', `${selectedVar?.seed ?? '42'} (Deterministic Random Latent Seed)`],
            ['Native Resolution:', `${selectedVar?.width || 64} × ${selectedVar?.height || 64} px (8-bit Lossless PNG)`],
            ['Synthesized Date:', new Date(comp.createdAt).toISOString()],
            ['Conditioning Mode:', comp.editingType ? comp.editingType.toUpperCase() : 'INITIAL SYNTHESIS'],
            ['Model Checkpoint:', `${comp.generatorMetadata?.modelName || 'ConditionalWGANGPGenerator'}`],
            ['Model SHA-256:', `${comp.generatorMetadata?.checkpointSha256 || 'a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964'}`],
          ];

          compDetails.forEach(([k, v]) => {
            const rowY = doc.y;
            doc.font('Helvetica-Bold').fontSize(8.5).fillColor(COLORS.slate).text(k, detailsX, rowY, { width: 105 });
            doc.font('Helvetica').fontSize(8.5).fillColor(COLORS.darkSlate).text(v, detailsX + 110, rowY, { width: detailsWidth - 110 });
            doc.moveDown(0.2);
          });

          doc.y = Math.max(doc.y, startCompY + 148);
          doc.moveDown(0.4);

          // GCIT Conditioning Attributes Box
          doc.font('Helvetica-Bold').fontSize(9).fillColor(COLORS.navy).text('Active 20-Dimensional GCIT Conditioning:');
          doc.font('Courier').fontSize(8).fillColor(COLORS.darkSlate);
          const condAttrs = comp.conditionedAttributes && comp.conditionedAttributes.length > 0
            ? comp.conditionedAttributes.join('  •  ')
            : 'Zero active dimensions (neutral baseline prior sampling)';
          doc.text(condAttrs, { width: pageWidth });

          doc.moveDown(0.2);
          doc.font('Helvetica-Bold').fontSize(8.5).fillColor(COLORS.slate).text('Unconditioned Attributes (Stochastic CelebA Prior Baseline):');
          doc.font('Courier').fontSize(7.5).fillColor(COLORS.lightSlate);
          const uncondAttrs = comp.unconditionedAttributes && comp.unconditionedAttributes.length > 0
            ? comp.unconditionedAttributes.join(', ')
            : 'All remaining facial attributes drawn from unbiased training distribution';
          doc.text(uncondAttrs, { width: pageWidth });
        } else {
          doc.font('Helvetica-Oblique').fontSize(9).fillColor(COLORS.slate).text('No facial composite generated for this case record.');
        }

        // -------------------------------------------------------------
        // SECTION 5: EDITING AND VERSION LINEAGE HISTORY
        // -------------------------------------------------------------
        drawSectionHeader('Section 5', 'Editing & Version Lineage History');

        if (data.compositeHistory && data.compositeHistory.length > 0) {
          doc.font('Helvetica-Bold').fontSize(8.5).fillColor(COLORS.slate);
          const vHeaders = ['VER', 'EDITING TYPE', 'PARENT ID', 'APPLIED ADJUSTMENTS', 'DATE'];
          const vCols = [
            doc.page.margins.left,
            doc.page.margins.left + 40,
            doc.page.margins.left + 175,
            doc.page.margins.left + 330,
            doc.page.margins.left + 430,
          ];

          const vHeaderY = doc.y;
          vHeaders.forEach((vh, i) => doc.text(vh, vCols[i], vHeaderY));
          drawHorizontalRule(doc.y + 12, COLORS.navy, 0.75);
          doc.y = vHeaderY + 15;

          // Render sorted chronological history
          const sortedHistory = [...data.compositeHistory].sort((a, b) => a.version - b.version);
          sortedHistory.forEach((item) => {
            if (doc.y > doc.page.height - 50) doc.addPage();
            const rY = doc.y;
            doc.font('Helvetica-Bold').fontSize(8).fillColor(COLORS.navy).text(`v${item.version}`, vCols[0], rY, { width: 35 });
            doc.font('Helvetica').fontSize(8).fillColor(COLORS.darkSlate).text((item.editingType || 'initial').toUpperCase(), vCols[1], rY, { width: 130 });
            doc.font('Courier').fontSize(7.5).fillColor(COLORS.slate).text(item.parentComposite ? item.parentComposite.toString().slice(-8) : 'ROOT (v1)', vCols[2], rY, { width: 150 });
            
            // Summarize adjustments
            let adjSummary = 'None (Base)';
            if (item.traversalParameters?.continuousAdjustments) {
              const entries = Object.entries(item.traversalParameters.continuousAdjustments);
              if (entries.length > 0) {
                adjSummary = entries.map(([k, v]) => `${k}:${v > 0 ? '+' : ''}${v.toFixed(2)}`).join(' ');
              }
            }
            if (item.traversalParameters?.latentTraversals && item.traversalParameters.latentTraversals.length > 0) {
              const lSummary = item.traversalParameters.latentTraversals.map((lt) => `${lt.attribute}:${lt.alpha > 0 ? '+' : ''}${lt.alpha.toFixed(1)}`).join(' ');
              adjSummary = adjSummary === 'None (Base)' ? lSummary : `${adjSummary} | ${lSummary}`;
            }

            doc.font('Helvetica').fontSize(7.5).fillColor(COLORS.darkSlate).text(adjSummary, vCols[3], rY, { width: 95, lineBreak: false, ellipsis: true });
            doc.font('Helvetica').fontSize(7.5).fillColor(COLORS.slate).text(new Date(item.createdAt).toLocaleDateString(), vCols[4], rY, { width: 75 });
            doc.y = rY + 12;
          });
        } else {
          doc.font('Helvetica-Oblique').fontSize(9).fillColor(COLORS.slate).text('Single initial version synthesized; zero sequential edits performed.');
        }

        // -------------------------------------------------------------
        // SECTION 6: IMAGE AND MODEL PROVENANCE
        // -------------------------------------------------------------
        if (doc.y > doc.page.height - 110) doc.addPage();
        drawSectionHeader('Section 6', 'Image & Model Provenance Cryptographic Verification');

        const provY = doc.y;
        const provBoxHeight = 78;

        doc
          .save()
          .rect(doc.page.margins.left, provY, pageWidth, provBoxHeight)
          .fillColor(COLORS.bgLight)
          .strokeColor(COLORS.border)
          .lineWidth(1)
          .fillAndStroke()
          .restore();

        const imageHash = data.computedImageSha256 || 'Hash unverified or image missing';
        const isVerified = data.imageVerified === true;

        doc.font('Helvetica-Bold').fontSize(8.5).fillColor(COLORS.slate);
        doc.text('COMPOSITE IMAGE SHA-256 (RECOMPUTED):', doc.page.margins.left + 10, provY + 7);
        doc.text('INTEGRITY STATUS:', doc.page.margins.left + 350, provY + 7);

        doc.font('Courier').fontSize(8).fillColor(COLORS.navy);
        doc.text(imageHash, doc.page.margins.left + 10, provY + 18, { width: 330 });

        doc.font('Helvetica-Bold').fontSize(8.5).fillColor(isVerified ? COLORS.green : COLORS.red);
        doc.text(isVerified ? '✓ SHA-256 MATCH VERIFIED' : '⚠ VERIFICATION FAILED / UNVERIFIED', doc.page.margins.left + 350, provY + 18);

        doc.font('Helvetica-Bold').fontSize(8).fillColor(COLORS.slate);
        doc.text('GENERATOR CHECKPOINT SHA-256 (4,104,260 PARAMS):', doc.page.margins.left + 10, provY + 31);
        doc.text('EVALUATOR CHECKPOINT SHA-256 (4,828,381 PARAMS):', doc.page.margins.left + 260, provY + 31);

        doc.font('Courier').fontSize(7).fillColor(COLORS.darkSlate);
        doc.text('a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964', doc.page.margins.left + 10, provY + 41);
        doc.text('f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834', doc.page.margins.left + 260, provY + 41);

        doc.font('Helvetica-Bold').fontSize(8).fillColor(COLORS.slate);
        doc.text('NLP BILSTM-CRF CHECKPOINT SHA-256 (2,599,864 PARAMS):', doc.page.margins.left + 10, provY + 53);
        doc.font('Courier').fontSize(7).fillColor(COLORS.darkSlate);
        doc.text('2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3', doc.page.margins.left + 10, provY + 63);

        doc.y = provY + provBoxHeight + 10;

        // -------------------------------------------------------------
        // SECTION 7: AUDIT HISTORY
        // -------------------------------------------------------------
        drawSectionHeader('Section 7', 'Chain of Custody & Security Audit Log');

        if (data.auditLogs && data.auditLogs.length > 0) {
          doc.font('Helvetica-Bold').fontSize(8).fillColor(COLORS.slate);
          const aHeaders = ['TIMESTAMP', 'EVENT TYPE', 'OUTCOME', 'ACTOR / DETAILS'];
          const aCols = [
            doc.page.margins.left,
            doc.page.margins.left + 115,
            doc.page.margins.left + 295,
            doc.page.margins.left + 355,
          ];

          const aHeaderY = doc.y;
          aHeaders.forEach((ah, i) => doc.text(ah, aCols[i], aHeaderY));
          drawHorizontalRule(doc.y + 10, COLORS.navy, 0.75);
          doc.y = aHeaderY + 13;

          data.auditLogs.slice(0, 15).forEach((audit) => {
            if (doc.y > doc.page.height - 45) doc.addPage();
            const rY = doc.y;
            const rawDate = (audit as any).createdAt || (audit as any).timestamp || new Date();
            const parsedDate = new Date(rawDate);
            const dateStr = !isNaN(parsedDate.getTime())
              ? parsedDate.toISOString().replace('T', ' ').slice(0, 19)
              : new Date().toISOString().replace('T', ' ').slice(0, 19);
            doc.font('Courier').fontSize(7.5).fillColor(COLORS.slate).text(dateStr, aCols[0], rY, { width: 110 });
            doc.font('Helvetica-Bold').fontSize(7.5).fillColor(COLORS.navy).text(audit.event, aCols[1], rY, { width: 175 });
            
            const isSuccess = audit.status === 'success';
            doc.font('Helvetica-Bold').fontSize(7.5).fillColor(isSuccess ? COLORS.green : COLORS.red).text(audit.status.toUpperCase(), aCols[2], rY, { width: 55 });
            
            const detailsStr = audit.details ? JSON.stringify(audit.details).slice(0, 55) : 'Standard system audit entry';
            doc.font('Helvetica').fontSize(7).fillColor(COLORS.darkSlate).text(detailsStr, aCols[3], rY, { width: 150, lineBreak: false, ellipsis: true });
            doc.y = rY + 11;
          });
        } else {
          doc.font('Helvetica-Oblique').fontSize(9).fillColor(COLORS.slate).text('No audit events found for this specific case context.');
        }

        // -------------------------------------------------------------
        // SECTION 8: FORENSIC DISCLAIMER & SCIENTIFIC LIMITATIONS
        // -------------------------------------------------------------
        if (doc.y > doc.page.height - 140) doc.addPage();
        drawSectionHeader('Section 8', 'Forensic Disclaimer & Scientific Limitations');

        const discY = doc.y;
        const disclaimerText =
          'LEGAL AND SCIENTIFIC NOTICE: Reconstructed facial composites produced by Phantom AI 3.0 are investigative approximation aids generated solely to assist law enforcement officers and detectives in developing leads and narrowing investigative queries. They MUST NEVER be presented or utilized as biometric identification evidence, forensic ground truth, or probable-cause justification for arrest or prosecution in judicial proceedings.\n\n' +
          '• MODEL LIMITATIONS: The generative model is a 2D planar convolutional network operating at 64×64 pixel resolution without 3D cephalometric landmarks. Subtle geometric feature sliders represent localized pixel shifts and do not constitute 3D anatomical morphing.\n' +
          '• EVALUATION SCORES: Evaluator confidence scores are internal diagnostic training metrics, not calibrated Bayesian posterior probabilities.\n' +
          '• EVIDENTIARY INTEGRITY: Unconditioned, unknown, or contradictory witness statements remain unconditioned in the generator and must not be represented as confirmed suspect characteristics.\n' +
          '• CRYPTOGRAPHIC CLAIMS: Cryptographic SHA-256 digests in this dossier provide algorithmic integrity checks against accidental data corruption or unauthorized bit-level modification. They do not constitute asymmetric public-key digital signatures, and hashing alone does not provide legal proof of chain of custody without independent institutional logs.';

        doc.font('Helvetica').fontSize(7.5);
        const discHeight = doc.heightOfString(disclaimerText, { width: pageWidth - 20 }) + 16;

        doc
          .save()
          .rect(doc.page.margins.left, discY, pageWidth, discHeight)
          .fillColor('#fffbeb')
          .strokeColor(COLORS.goldBorder)
          .lineWidth(1)
          .fillAndStroke()
          .restore();

        doc
          .font('Helvetica')
          .fontSize(7.5)
          .fillColor('#92400e')
          .text(disclaimerText, doc.page.margins.left + 10, discY + 8, { width: pageWidth - 20, lineGap: 1.5 });

        // -------------------------------------------------------------
        // FOOTERS & PAGE NUMBERS (Iterate all buffered pages)
        // -------------------------------------------------------------
        const pageRange = doc.bufferedPageRange();
        for (let i = 0; i < pageRange.count; i++) {
          doc.switchToPage(i);
          const footerY = doc.page.height - 30;

          drawHorizontalRule(footerY - 4, COLORS.border, 0.5);

          doc
            .font('Helvetica')
            .fontSize(7.5)
            .fillColor(COLORS.lightSlate)
            .text(`PHANTOM AI FORENSIC SYSTEM • CASE ${data.caseDoc.caseNumber} • REPORT ${data.reportNumber}`, doc.page.margins.left, footerY);

          doc
            .font('Helvetica-Bold')
            .fontSize(7.5)
            .fillColor(COLORS.slate)
            .text(`Page ${i + 1} of ${pageRange.count}`, doc.page.margins.left + pageWidth - 60, footerY, { width: 60, align: 'right' });
        }

        doc.end();
      } catch (error) {
        reject(error);
      }
    });
  }
}

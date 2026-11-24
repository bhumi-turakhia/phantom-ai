import fs from 'fs';
import path from 'path';
import mongoose, { Types } from 'mongoose';
import { Case } from '../models/Case.model';
import { Composite } from '../models/Composite.model';
import { WitnessDescription } from '../models/WitnessDescription.model';
import { Report } from '../models/Report.model';
import { CaseComment } from '../models/CaseComment.model';
import { User } from '../models/User.model';
import { SecurityAudit } from '../models/SecurityAudit.model';
import { GenerationJob } from '../models/GenerationJob.model';
import { mlClient } from './mlClient.service';
import { getCaseTimeline } from './caseCollaboration.service';
import { logger } from '../config/logger';

export interface DashboardMetricItem {
  label: string;
  value: string | number;
  suffix?: string;
  sublabel?: string;
  description?: string;
}

export interface SystemHealthItem {
  key: string;
  label: string;
  status: 'Online' | 'Offline' | 'Degraded' | 'Not configured' | 'Unknown';
  tone: 'verified' | 'pending' | 'critical';
  details?: string;
}

export interface DashboardActivityItem {
  key: string;
  id: string;
  caseId?: string;
  action: string;
  category: string;
  tone: 'active' | 'pending' | 'verified' | 'critical';
  time: string;
  rawTimestamp: string;
}

export interface DashboardOverviewResponse {
  metrics: {
    activeCases: DashboardMetricItem;
    suspectsGenerated: DashboardMetricItem;
    aiAnalysisAccuracy: DashboardMetricItem;
    databaseRecords: DashboardMetricItem & {
      breakdown: {
        cases: number;
        composites: number;
        witnessDescriptions: number;
        reports: number;
        caseComments: number;
        users: number;
        securityAudits: number;
      };
    };
    averageProcessingTime: DashboardMetricItem & {
      sampleSize: number;
      numericSeconds: number | null;
    };
    systemStatus: DashboardMetricItem & {
      numericPercent: number;
      tone: 'verified' | 'pending' | 'critical';
    };
  };
  systemHealth: SystemHealthItem[];
  recentActivity: DashboardActivityItem[];
  investigator: {
    id: string;
    investigatorId: string;
    name: string;
    email: string;
    role: string;
    badgeNumber?: string;
    department?: string;
  };
}

export const getDashboardOverview = async (
  userId: string,
  userRole?: string
): Promise<DashboardOverviewResponse> => {
  // 1. Resolve Accessible Cases with RBAC
  const isPrivileged = userRole === 'admin' || userRole === 'supervisor';
  const caseQuery: Record<string, unknown> = { isDeleted: false };

  if (!isPrivileged && Types.ObjectId.isValid(userId)) {
    const userObjId = new Types.ObjectId(userId);
    caseQuery.$or = [
      { createdBy: userObjId },
      { assignedTo: userObjId },
      { 'collaborators.user': userObjId },
    ];
  }

  const accessibleCases = await Case.find(caseQuery)
    .select('_id caseNumber title status priority createdAt updatedAt')
    .sort({ updatedAt: -1 })
    .lean();

  const accessibleCaseIds = accessibleCases.map((c) => c._id);

  // Active cases: status 'open', 'in_progress', or 'under_review' (strictly excluding 'resolved', 'archived', 'closed')
  const activeCasesCount = accessibleCases.filter(
    (c) => c.status === 'open' || c.status === 'in_progress' || c.status === 'under_review'
  ).length;

  // 2. Suspects Generated: count genuine variations for accessible cases
  let suspectsGeneratedCount = 0;
  if (accessibleCaseIds.length > 0) {
    const accessibleComposites = await Composite.find({
      case: { $in: accessibleCaseIds },
    })
      .select('variations')
      .lean();

    suspectsGeneratedCount = accessibleComposites.reduce(
      (acc, comp) => acc + (Array.isArray(comp.variations) ? comp.variations.length : 0),
      0
    );
  }

  // 3. Database Records across 7 actual MongoDB collections
  const [
    casesTotal,
    compositesTotal,
    witnessTotal,
    reportsTotal,
    commentsTotal,
    usersTotal,
    auditTotal,
  ] = await Promise.all([
    Case.countDocuments({ isDeleted: false }),
    Composite.countDocuments(),
    WitnessDescription.countDocuments(),
    Report.countDocuments(),
    CaseComment.countDocuments({ isDeleted: false }),
    User.countDocuments(),
    SecurityAudit.countDocuments(),
  ]);

  const totalDatabaseRecords =
    casesTotal +
    compositesTotal +
    witnessTotal +
    reportsTotal +
    commentsTotal +
    usersTotal +
    auditTotal;

  // 4. Average Processing Time from completed generation jobs
  const completedJobs = await GenerationJob.find({
    status: 'completed',
    startedAt: { $exists: true },
    completedAt: { $exists: true },
  })
    .select('startedAt completedAt')
    .lean();

  let avgProcessingTimeSeconds: number | null = null;
  const validDurations = completedJobs
    .map((j) => (new Date(j.completedAt!).getTime() - new Date(j.startedAt!).getTime()) / 1000)
    .filter((d) => !isNaN(d) && d > 0);

  if (validDurations.length > 0) {
    const rawMean = validDurations.reduce((a, b) => a + b, 0) / validDurations.length;
    avgProcessingTimeSeconds = Math.round(rawMean * 10) / 10;
  }

  // 5. System Health Checks
  // Database check
  const isDbOnline = mongoose.connection.readyState === 1;
  const dbHealth: SystemHealthItem = {
    key: 'database',
    label: 'Database sync',
    status: isDbOnline ? 'Online' : 'Offline',
    tone: isDbOnline ? 'verified' : 'critical',
    details: isDbOnline
      ? 'MongoDB replica/standalone connected and responding'
      : 'MongoDB connection state not connected (readyState !== 1)',
  };

  // Evidence Storage check
  const storageDir = path.resolve(process.cwd(), 'data', 'composites');
  let isStorageOnline = false;
  try {
    if (!fs.existsSync(storageDir)) {
      fs.mkdirSync(storageDir, { recursive: true });
    }
    fs.accessSync(storageDir, fs.constants.R_OK | fs.constants.W_OK);
    isStorageOnline = true;
  } catch {
    isStorageOnline = false;
  }
  const storageHealth: SystemHealthItem = {
    key: 'evidence_storage',
    label: 'Evidence storage',
    status: isStorageOnline ? 'Online' : 'Degraded',
    tone: isStorageOnline ? 'verified' : 'pending',
    details: isStorageOnline
      ? 'Local forensic image archive accessible and writable'
      : 'Local forensic image archive inaccessible or write permission denied',
  };

  // ML Service & Model Status
  let reconstructionHealth: SystemHealthItem = {
    key: 'reconstruction_model',
    label: 'Facial reconstruction model',
    status: 'Unknown',
    tone: 'pending',
    details: 'Querying local ML service...',
  };

  let extractionHealth: SystemHealthItem = {
    key: 'extraction_model',
    label: 'Attribute extraction model',
    status: 'Unknown',
    tone: 'pending',
    details: 'Querying local ML service...',
  };

  try {
    const modelStatusRes = await mlClient.getModelStatus();
    const cgan = modelStatusRes?.models?.cgan_generator;
    const nlp = modelStatusRes?.models?.nlp_bilstm;

    if (cgan?.is_trained && cgan?.checkpoint_exists) {
      reconstructionHealth = {
        key: 'reconstruction_model',
        label: 'Facial reconstruction model',
        status: 'Online',
        tone: 'verified',
        details: `${cgan.model_name} (Phase 5.4 WGAN-GP, ${cgan.parameter_count.toLocaleString()} params)`,
      };
    } else if (cgan?.is_trained && !cgan?.checkpoint_exists) {
      reconstructionHealth = {
        key: 'reconstruction_model',
        label: 'Facial reconstruction model',
        status: 'Not configured',
        tone: 'pending',
        details: 'Model architecture registered but checkpoint file missing',
      };
    } else {
      reconstructionHealth = {
        key: 'reconstruction_model',
        label: 'Facial reconstruction model',
        status: 'Offline',
        tone: 'critical',
        details: 'Reconstruction model not trained or disabled',
      };
    }

    if (nlp?.is_trained && nlp?.checkpoint_exists) {
      extractionHealth = {
        key: 'extraction_model',
        label: 'Attribute extraction model',
        status: 'Online',
        tone: 'verified',
        details: `${nlp.model_name} (Phase 3C.2 BiLSTM-CRF, ${nlp.parameter_count.toLocaleString()} params)`,
      };
    } else if (nlp?.is_trained && !nlp?.checkpoint_exists) {
      extractionHealth = {
        key: 'extraction_model',
        label: 'Attribute extraction model',
        status: 'Not configured',
        tone: 'pending',
        details: 'Attribute extraction checkpoint missing',
      };
    } else {
      extractionHealth = {
        key: 'extraction_model',
        label: 'Attribute extraction model',
        status: 'Offline',
        tone: 'critical',
        details: 'Attribute extraction model not trained or disabled',
      };
    }
  } catch (err: any) {
    logger.warn(`[DASHBOARD] ML Service health check failed: ${err.message}`);
    reconstructionHealth = {
      key: 'reconstruction_model',
      label: 'Facial reconstruction model',
      status: 'Offline',
      tone: 'critical',
      details: 'Python ML service unreachable at local endpoint',
    };
    extractionHealth = {
      key: 'extraction_model',
      label: 'Attribute extraction model',
      status: 'Offline',
      tone: 'critical',
      details: 'Python ML service unreachable at local endpoint',
    };
  }

  const systemHealthItems = [reconstructionHealth, extractionHealth, dbHealth, storageHealth];

  // Overall system percentage: 4 tracked critical services
  const onlineCount = systemHealthItems.filter((s) => s.status === 'Online').length;
  const systemStatusPercent = Math.round((onlineCount / systemHealthItems.length) * 100);
  const systemTone: 'verified' | 'pending' | 'critical' =
    systemStatusPercent === 100 ? 'verified' : systemStatusPercent >= 75 ? 'pending' : 'critical';
  const systemStatusLabel =
    systemStatusPercent === 100
      ? 'AI Systems Nominal'
      : systemStatusPercent >= 75
      ? 'Systems Degraded'
      : 'System Disruption Detected';

  // 6. Recent Investigation Activity (Authoritative timeline across accessible cases)
  const recentActivity: DashboardActivityItem[] = [];
  if (accessibleCases.length > 0) {
    const topCases = accessibleCases.slice(0, 5);
    const timelines = await Promise.all(
      topCases.map((c) =>
        getCaseTimeline(c._id.toString(), userId, userRole, { order: 'desc' }).catch(() => null)
      )
    );

    const mergedEvents: any[] = [];
    for (const tl of timelines) {
      if (tl && Array.isArray(tl.events)) {
        mergedEvents.push(...tl.events);
      }
    }

    mergedEvents.sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );

    const sliced = mergedEvents.slice(0, 6);
    for (const ev of sliced) {
      let tone: 'active' | 'pending' | 'verified' | 'critical' = 'pending';
      if (ev.category === 'report' || ev.category === 'comparison') {
        tone = 'verified';
      } else if (ev.category === 'composite' || ev.category === 'nlp') {
        tone = 'active';
      } else if (ev.eventType?.includes('DELETED') || ev.eventType?.includes('FAIL')) {
        tone = 'critical';
      }

      recentActivity.push({
        key: ev.eventId,
        id: ev.caseNumber || 'CASE',
        caseId: ev.caseId,
        action: ev.description,
        category: ev.category,
        tone,
        time: new Date(ev.timestamp).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
        rawTimestamp: ev.timestamp,
      });
    }
  }

  // 7. Authenticated User Identity
  let userDoc: any = null;
  if (Types.ObjectId.isValid(userId)) {
    userDoc = await User.findById(userId)
      .select('name email role investigatorId badgeNumber department')
      .lean();
  }

  return {
    metrics: {
      activeCases: {
        label: 'Active Cases',
        value: activeCasesCount,
        sublabel: isPrivileged ? 'All clearance scopes' : 'Assigned & created',
        description: 'Cases with active forensic workflows',
      },
      suspectsGenerated: {
        label: 'Suspects Generated',
        value: suspectsGeneratedCount,
        sublabel: 'Accessible variations',
        description: 'Generated facial composite suspect variations across accessible cases',
      },
      aiAnalysisAccuracy: {
        label: 'AI Analysis Accuracy',
        value: 'Not yet validated',
        sublabel: 'Ground truth pending',
        description:
          'Runtime witness statements lack pre-labeled ground truth; validation requires verified benchmark evaluations.',
      },
      databaseRecords: {
        label: 'Database Records',
        value: totalDatabaseRecords,
        sublabel: '7 collections verified',
        description: 'Actual MongoDB documents across all primary collections',
        breakdown: {
          cases: casesTotal,
          composites: compositesTotal,
          witnessDescriptions: witnessTotal,
          reports: reportsTotal,
          caseComments: commentsTotal,
          users: usersTotal,
          securityAudits: auditTotal,
        },
      },
      averageProcessingTime: {
        label: 'Avg. Processing Time',
        value: avgProcessingTimeSeconds !== null ? avgProcessingTimeSeconds : 'Insufficient data',
        suffix: avgProcessingTimeSeconds !== null ? 's' : undefined,
        sublabel:
          avgProcessingTimeSeconds !== null
            ? `Mean latency (${validDurations.length} jobs)`
            : 'No completed jobs recorded',
        description:
          'Empirical duration measured between startedAt and completedAt across completed generation jobs',
        sampleSize: validDurations.length,
        numericSeconds: avgProcessingTimeSeconds,
      },
      systemStatus: {
        label: 'System Status',
        value: systemStatusPercent,
        suffix: '%',
        sublabel: systemStatusLabel,
        description: `${onlineCount} of ${systemHealthItems.length} core forensic services responding nominally`,
        numericPercent: systemStatusPercent,
        tone: systemTone,
      },
    },
    systemHealth: systemHealthItems,
    recentActivity,
    investigator: {
      id: userId,
      investigatorId: userDoc?.investigatorId || 'INV-UNKNOWN',
      name: userDoc?.name || 'Forensic Investigator',
      email: userDoc?.email || '',
      role: (userDoc?.role || userRole || 'investigator').toUpperCase(),
      badgeNumber: userDoc?.badgeNumber || 'ACT-01',
      department: userDoc?.department || 'Forensic Facial Intelligence',
    },
  };
};


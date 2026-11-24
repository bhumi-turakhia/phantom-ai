import api from './api';

export interface DashboardMetric {
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

export interface DashboardOverviewData {
  metrics: {
    activeCases: DashboardMetric;
    suspectsGenerated: DashboardMetric;
    aiAnalysisAccuracy: DashboardMetric;
    databaseRecords: DashboardMetric & {
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
    averageProcessingTime: DashboardMetric & {
      sampleSize: number;
      numericSeconds: number | null;
    };
    systemStatus: DashboardMetric & {
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

export const getDashboardOverview = async (): Promise<DashboardOverviewData> => {
  const res = await api.get('/dashboard/overview');
  return res.data?.data;
};


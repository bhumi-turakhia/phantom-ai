import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  FolderSearch,
  ScanFace,
  Gauge,
  Database,
  Clock,
  Activity,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { StatusBadge } from "../components/ui/Badge";
import { EvidenceTag } from "../components/ui/EvidenceTag";
import { PageTransition } from "../components/layout/PageTransition";
import {
  getDashboardOverview,
  type DashboardOverviewData,
} from "../services/dashboardService";

export default function Dashboard() {
  const [data, setData] = useState<DashboardOverviewData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const overview = await getDashboardOverview();
      setData(overview);
    } catch (err: any) {
      const msg =
        err?.response?.data?.message ||
        err?.message ||
        "Failed to load operational telemetry.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Format StatCard data from genuine backend response
  const metrics = [
    {
      label: "Active Cases",
      value: data?.metrics?.activeCases?.value ?? (isLoading ? "..." : 0),
      sublabel: data?.metrics?.activeCases?.sublabel || "Active investigations",
      icon: <FolderSearch className="h-4 w-4" />,
    },
    {
      label: "Suspects Generated",
      value: data?.metrics?.suspectsGenerated?.value ?? (isLoading ? "..." : 0),
      sublabel: data?.metrics?.suspectsGenerated?.sublabel || "Verified variations",
      icon: <ScanFace className="h-4 w-4" />,
    },
    {
      label: "AI Analysis Accuracy",
      value: data?.metrics?.aiAnalysisAccuracy?.value ?? (isLoading ? "..." : "Not yet validated"),
      sublabel: data?.metrics?.aiAnalysisAccuracy?.sublabel || "Ground truth pending",
      icon: <Gauge className="h-4 w-4" />,
    },
    {
      label: "Database Records",
      value: data?.metrics?.databaseRecords?.value ?? (isLoading ? "..." : 0),
      sublabel: data?.metrics?.databaseRecords?.sublabel || "MongoDB verified",
      icon: <Database className="h-4 w-4" />,
    },
    {
      label: "Avg. Processing Time",
      value:
        data?.metrics?.averageProcessingTime?.value !== undefined
          ? data.metrics.averageProcessingTime.value
          : isLoading
          ? "..."
          : "Insufficient data",
      suffix: data?.metrics?.averageProcessingTime?.suffix,
      sublabel:
        data?.metrics?.averageProcessingTime?.sublabel || "Generation duration",
      icon: <Clock className="h-4 w-4" />,
    },
    {
      label: "System Status",
      value: data?.metrics?.systemStatus?.value ?? (isLoading ? "..." : 0),
      suffix: data?.metrics?.systemStatus?.suffix || "%",
      sublabel: data?.metrics?.systemStatus?.sublabel || "Operational telemetry",
      icon: <Activity className="h-4 w-4" />,
    },
  ];

  const systemHealth = data?.systemHealth || [
    {
      key: "reconstruction_model",
      label: "Facial reconstruction model",
      status: "Offline" as const,
      tone: "critical" as const,
      details: "Awaiting telemetry",
    },
    {
      key: "extraction_model",
      label: "Attribute extraction model",
      status: "Offline" as const,
      tone: "critical" as const,
      details: "Awaiting telemetry",
    },
    {
      key: "database",
      label: "Database sync",
      status: "Offline" as const,
      tone: "critical" as const,
      details: "Awaiting telemetry",
    },
    {
      key: "evidence_storage",
      label: "Evidence storage",
      status: "Offline" as const,
      tone: "critical" as const,
      details: "Awaiting telemetry",
    },
  ];

  const recentActivity = data?.recentActivity || [];
  const overallTone = data?.metrics?.systemStatus?.tone || (isLoading ? "pending" : "critical");
  const overallLabel =
    data?.metrics?.systemStatus?.sublabel || (isLoading ? "Verifying Systems..." : "Telemetry Unavailable");

  return (
    <PageTransition>
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold text-ink-primary">Operations Overview</h1>
            <p className="label-technical text-ink-disabled mt-1">
              Live forensic intelligence telemetry & system status
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={fetchDashboardData}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded border border-line bg-bg-surface hover:bg-bg-inset text-ink-secondary transition-colors cursor-pointer"
              title="Refresh Telemetry"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
              <span className="font-mono text-[11px]">Sync</span>
            </button>
            <StatusBadge tone={overallTone}>{overallLabel}</StatusBadge>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-3 p-3.5 rounded border border-[#D64550]/30 bg-[#D64550]/10 text-[#D64550] text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <div className="flex-1">
              <strong>Telemetry Error:</strong> {error}
            </div>
            <button
              type="button"
              onClick={fetchDashboardData}
              className="underline font-semibold hover:text-white"
            >
              Retry
            </button>
          </div>
        )}

        {/* 6 Core Operational Telemetry Stat Cards */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
          {metrics.map((m, i) => (
            <StatCard key={m.label} {...m} delay={i * 0.05} />
          ))}
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Investigation Activity Timeline */}
          <Card className="lg:col-span-2" delay={0.15}>
            <CardHeader>
              <CardTitle>Investigation Activity</CardTitle>
              <span className="label-technical text-ink-disabled">
                {recentActivity.length > 0
                  ? "Live case timeline feed"
                  : "Verified case activity"}
              </span>
            </CardHeader>
            <CardContent className="space-y-3">
              {recentActivity.length > 0 ? (
                recentActivity.map((item, i) => (
                  <motion.div
                    key={item.key}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.15 + i * 0.04 }}
                    className="flex items-center justify-between gap-3 rounded border border-line bg-bg-inset px-4 py-3"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <EvidenceTag caseId={item.id} />
                      <span className="text-sm text-ink-secondary truncate" title={item.action}>
                        {item.action}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <StatusBadge tone={item.tone}>{item.tone}</StatusBadge>
                      <span className="text-xs font-mono text-ink-disabled">{item.time}</span>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center py-12 px-4 text-center rounded border border-line/60 bg-bg-inset/50">
                  <FolderSearch className="w-8 h-8 text-ink-disabled/60 mb-2.5" />
                  <p className="text-sm font-semibold text-ink-secondary">
                    No recent activity
                  </p>
                  <p className="text-xs text-ink-disabled max-w-sm mt-1">
                    No timeline actions or intelligence events recorded for your accessible cases.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Real Live System Health Status */}
          <Card delay={0.2}>
            <CardHeader>
              <CardTitle>System Status</CardTitle>
              <span className="label-technical text-ink-disabled">
                Live Subsystem Health
              </span>
            </CardHeader>
            <CardContent className="space-y-4">
              {systemHealth.map((s) => (
                <div
                  key={s.label}
                  className="group flex flex-col gap-1 rounded border border-transparent hover:border-line hover:bg-bg-inset/40 p-1.5 -mx-1.5 transition-colors"
                >
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-ink-secondary font-medium">{s.label}</span>
                    <StatusBadge tone={s.tone}>{s.status}</StatusBadge>
                  </div>
                  {s.details && (
                    <span
                      className="text-[11px] font-mono text-ink-disabled truncate"
                      title={s.details}
                    >
                      {s.details}
                    </span>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </PageTransition>
  );
}

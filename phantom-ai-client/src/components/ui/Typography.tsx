import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/utils";

/**
 * Standardized top-of-page header for all Phantom AI 3.0 workspace views.
 */
export function PageHeader({
  title,
  subtitle,
  code,
  actions,
  className,
}: {
  title: string;
  subtitle?: string;
  code?: string;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-start justify-between gap-4 border-b border-[#DCE4EE] pb-4",
        className
      )}
    >
      <div className="space-y-1">
        <div className="flex items-center gap-2.5 flex-wrap">
          {code && (
            <span className="rounded bg-[#18263A] px-2 py-0.5 font-mono text-[10px] font-semibold tracking-wider text-[#27B3A3] uppercase">
              {code}
            </span>
          )}
          <h1 className="text-lg font-semibold tracking-tight text-[#18263A]">
            {title}
          </h1>
        </div>
        {subtitle && (
          <p className="text-xs text-[#5A718E] leading-relaxed">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/**
 * Standardized panel/section sub-header with optional technical badge and action slot.
 */
export function SectionHeader({
  title,
  subtitle,
  badge,
  actions,
  className,
}: {
  title: string;
  subtitle?: string;
  badge?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-between gap-2",
        className
      )}
    >
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-[#18263A]">{title}</h2>
          {badge}
        </div>
        {subtitle && <p className="text-xs text-[#5A718E] mt-0.5">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/**
 * Uppercase JetBrains Mono label for field headers, table columns, and telemetry labels.
 */
export function TechnicalLabel({
  children,
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn("label-technical text-[#5A718E]", className)}
      {...props}
    >
      {children}
    </span>
  );
}

/**
 * Inline JetBrains Mono readout for seeds, coordinates, timestamps, and GCIT values.
 */
export function MonoValue({
  children,
  tone = "default",
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & {
  tone?: "default" | "teal" | "amber" | "slate" | "critical";
}) {
  const toneClasses = {
    default: "text-[#18263A] bg-[#EEF3F8] border-[#DCE4EE]",
    teal: "text-[#178074] bg-[#E8F7F5] border-[#27B3A3]/30",
    amber: "text-[#9A6B16] bg-[#FDF7EB] border-[#D9A441]/40",
    slate: "text-[#3B4F6B] bg-[#F3F6FA] border-[#C1CFDF]",
    critical: "text-[#B92B36] bg-[#FDECEC] border-[#D64550]/30",
  }[tone];

  return (
    <code
      className={cn(
        "inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-xs font-medium",
        toneClasses,
        className
      )}
      {...props}
    >
      {children}
    </code>
  );
}

/**
 * Compact SHA-256 integrity hash pill rendered in JetBrains Mono.
 */
export function HashBadge({
  hash,
  prefixLength = 12,
  label = "SHA-256",
  className,
}: {
  hash: string;
  prefixLength?: number;
  label?: string;
  className?: string;
}) {
  const truncated =
    hash && hash.length > prefixLength ? `${hash.slice(0, prefixLength)}…` : hash;

  return (
    <span
      title={`${label}: ${hash}`}
      className={cn(
        "inline-flex items-center gap-1.5 rounded border border-[#DCE4EE] bg-[#EEF3F8] px-2 py-0.5 font-mono text-[11px] text-[#3B4F6B]",
        className
      )}
    >
      <span className="font-semibold text-[#7893B3]">{label}:</span>
      <span className="text-[#18263A]">{truncated || "N/A"}</span>
    </span>
  );
}


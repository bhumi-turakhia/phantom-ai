import React from "react";
import { cn } from "../../lib/utils";
import type { EvidenceEpistemicState } from "../../styles/tokens";

export type BadgeStatus =
  | "active"
  | "verified"
  | "critical"
  | "pending"
  | "neutral"
  | "info";

export interface StatusBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  status?: BadgeStatus;
  tone?: BadgeStatus;
  pulse?: boolean;
  dot?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  tone,
  pulse = false,
  dot = true,
  className,
  children,
  ...props
}) => {
  const resolvedStatus: BadgeStatus = status || tone || "neutral";

  const statusStyles: Record<BadgeStatus, { container: string; dot: string }> = {
    active: {
      container: "bg-[#27B3A3]/12 border-[#27B3A3]/35 text-[#147A6E]",
      dot: "bg-[#27B3A3]",
    },
    verified: {
      container: "bg-[#1D9E75]/12 border-[#1D9E75]/35 text-[#136F52]",
      dot: "bg-[#1D9E75]",
    },
    critical: {
      container: "bg-[#D64550]/12 border-[#D64550]/35 text-[#B4232E]",
      dot: "bg-[#D64550]",
    },
    pending: {
      container: "bg-[#D9A441]/15 border-[#D9A441]/40 text-[#966912]",
      dot: "bg-[#D9A441]",
    },
    neutral: {
      container: "bg-[#EEF3F8] border-[#C1CFDF] text-[#5A718E]",
      dot: "bg-[#7893B3]",
    },
    info: {
      container: "bg-[#18263A]/8 border-[#18263A]/20 text-[#18263A]",
      dot: "bg-[#18263A]",
    },
  };

  const style = statusStyles[resolvedStatus] || statusStyles.neutral;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[4px] border font-mono text-[10px] font-semibold uppercase tracking-[0.06em]",
        style.container,
        className
      )}
      {...props}
    >
      {dot && (
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full shrink-0",
            style.dot,
            pulse && "animate-pulse"
          )}
        />
      )}
      {children}
    </span>
  );
};

export interface EvidenceStateBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  state: EvidenceEpistemicState | string;
}

export const EvidenceStateBadge: React.FC<EvidenceStateBadgeProps> = ({
  state,
  className,
  ...props
}) => {
  const normalized = (state || "unknown").toLowerCase();

  const stateMap: Record<
    string,
    { label: string; classes: string; dot: string }
  > = {
    explicit: {
      label: "EXPLICIT · +1.0",
      classes: "bg-[#27B3A3]/12 border-[#27B3A3]/40 text-[#147A6E]",
      dot: "bg-[#27B3A3]",
    },
    present: {
      label: "PRESENT · +1.0",
      classes: "bg-[#27B3A3]/12 border-[#27B3A3]/40 text-[#147A6E]",
      dot: "bg-[#27B3A3]",
    },
    uncertain: {
      label: "UNCERTAIN · +0.5",
      classes: "bg-[#D9A441]/15 border-[#D9A441]/45 text-[#966912]",
      dot: "bg-[#D9A441]",
    },
    negated: {
      label: "NEGATED · 0.0",
      classes: "bg-[#18263A]/10 border-[#18263A]/30 text-[#18263A]",
      dot: "bg-[#18263A]",
    },
    absent: {
      label: "ABSENT · 0.0",
      classes: "bg-[#18263A]/10 border-[#18263A]/30 text-[#18263A]",
      dot: "bg-[#18263A]",
    },
    contradictory: {
      label: "CONTRADICTORY · MASKED",
      classes: "bg-[#D64550]/12 border-[#D64550]/40 text-[#B4232E]",
      dot: "bg-[#D64550]",
    },
    unknown: {
      label: "UNOBSERVED · MASK 0",
      classes: "bg-[#EEF3F8] border-[#C1CFDF] text-[#5A718E]",
      dot: "bg-[#7893B3]",
    },
  };

  const config = stateMap[normalized] || {
    label: state.toUpperCase(),
    classes: "bg-[#EEF3F8] border-[#C1CFDF] text-[#5A718E]",
    dot: "bg-[#7893B3]",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[4px] border font-mono text-[10px] font-semibold uppercase tracking-wider",
        config.classes,
        className
      )}
      {...props}
    >
      <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", config.dot)} />
      {config.label}
    </span>
  );
};

export const ClearanceBadge: React.FC<{
  level: string;
  className?: string;
}> = ({ level, className }) => (
  <span
    className={cn(
      "inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-[#18263A] border border-[#2A3B54] rounded-[4px] font-mono text-[10px] font-semibold text-[#27B3A3] tracking-[0.12em] uppercase",
      className
    )}
  >
    <span className="w-1.5 h-1.5 rounded-full bg-[#27B3A3]" />
    CLEARANCE: {level}
  </span>
);

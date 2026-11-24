import React from "react";
import { Card } from "./Card";
import { cn } from "../../lib/utils";

export interface StatCardProps {
  label: string;
  value: string | number;
  suffix?: string;
  sublabel?: string;
  icon?: React.ReactNode | React.ComponentType<{ className?: string }>;
  trend?: {
    value: string;
    direction: "up" | "down" | "neutral";
  };
  tone?: "default" | "teal" | "amber" | "slate";
  delay?: number;
  className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  suffix,
  sublabel,
  icon,
  trend,
  tone = "default",
  className,
}) => {
  const iconToneClasses = {
    default: "bg-[#EEF3F8] text-[#18263A] border-[#DCE4EE]",
    teal: "bg-[#27B3A3]/12 text-[#147A6E] border-[#27B3A3]/30",
    amber: "bg-[#D9A441]/15 text-[#966912] border-[#D9A441]/35",
    slate: "bg-[#18263A] text-[#27B3A3] border-[#2A3B54]",
  };

  const renderIcon = () => {
    if (!icon) return null;
    if (React.isValidElement(icon)) {
      return icon;
    }
    const IconComponent = icon as React.ComponentType<{ className?: string }>;
    return <IconComponent className="w-4 h-4" />;
  };

  return (
    <Card
      className={cn(
        "p-5 flex flex-col justify-between relative group hover:border-[#C1CFDF] transition-all",
        className
      )}
    >
      {/* Top accent line on hover */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-[#27B3A3] opacity-0 group-hover:opacity-100 transition-opacity" />

      <div className="flex items-start justify-between gap-3 mb-3">
        <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.08em] text-[#5A718E]">
          {label}
        </span>
        {icon && (
          <div
            className={cn(
              "w-8 h-8 rounded-[6px] border flex items-center justify-center shrink-0 transition-colors",
              iconToneClasses[tone]
            )}
          >
            {renderIcon()}
          </div>
        )}
      </div>

      <div className="space-y-1">
        <div className="font-mono text-2xl font-bold text-[#18263A] tracking-tight flex items-baseline gap-1">
          <span>{value}</span>
          {suffix && (
            <span className="text-sm font-mono font-medium text-[#7893B3]">
              {suffix}
            </span>
          )}
        </div>
        {(sublabel || trend) && (
          <div className="flex items-center justify-between gap-2 pt-1">
            {sublabel && (
              <div className="font-mono text-[11px] text-[#7893B3] truncate">
                {sublabel}
              </div>
            )}
            {trend && (
              <span
                className={cn(
                  "font-mono text-[10px] font-semibold px-1.5 py-0.5 rounded",
                  trend.direction === "up" && "bg-[#1D9E75]/12 text-[#136F52]",
                  trend.direction === "down" && "bg-[#D64550]/12 text-[#B4232E]",
                  trend.direction === "neutral" && "bg-[#EEF3F8] text-[#5A718E]"
                )}
              >
                {trend.value}
              </span>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};

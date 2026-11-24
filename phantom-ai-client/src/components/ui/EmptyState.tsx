import React from "react";
import { FingerprintMotif } from "./FingerprintMotif";
import { Button } from "./Button";
import { cn } from "../../lib/utils";

export interface EmptyStateProps {
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
  code?: string;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  actionLabel,
  onAction,
  icon,
  code = "NO_RECORDS_FOUND",
  className,
}) => (
  <div
    className={cn(
      "relative flex flex-col items-center justify-center p-12 text-center border border-dashed border-[#C1CFDF] rounded-md bg-white overflow-hidden",
      className
    )}
  >
    <FingerprintMotif opacity={0.04} className="w-60 h-60 text-[#18263A]" />
    <div className="relative z-10 max-w-md space-y-3">
      {icon && (
        <div className="mx-auto w-11 h-11 rounded-md bg-[#EEF3F8] border border-[#DCE4EE] flex items-center justify-center text-[#27B3A3]">
          {icon}
        </div>
      )}
      <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-[#7893B3]">
        [{code}]
      </div>
      <h3 className="font-display text-lg font-semibold text-[#18263A]">
        {title}
      </h3>
      <p className="text-xs text-[#5A718E] leading-relaxed">{description}</p>
      {actionLabel && onAction && (
        <div className="pt-2">
          <Button variant="primary" size="sm" onClick={onAction}>
            {actionLabel}
          </Button>
        </div>
      )}
    </div>
  </div>
);

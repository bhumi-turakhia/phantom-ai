import React from "react";
import { cn } from "../../lib/utils";

export interface EvidenceTagProps extends React.HTMLAttributes<HTMLDivElement> {
  id?: string;
  caseId?: string;
  label?: string;
}

export const EvidenceTag: React.FC<EvidenceTagProps> = ({
  id,
  caseId,
  label,
  className,
  ...props
}) => {
  const displayId = caseId || id || "UNASSIGNED";

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 px-2.5 py-1 bg-[#EEF3F8] border border-dashed border-[#C1CFDF] rounded-[4px]",
        className
      )}
      {...props}
    >
      <div className="w-1.5 h-1.5 rounded-full bg-[#27B3A3] shrink-0" />
      {label && (
        <span className="font-mono text-[10px] text-[#5A718E] uppercase tracking-wider">
          {label}:
        </span>
      )}
      <span className="font-mono text-[11px] font-medium text-[#18263A] tracking-wider">
        {displayId}
      </span>
    </div>
  );
};

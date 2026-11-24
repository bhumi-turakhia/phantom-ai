import React from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/utils";

export interface LoadingForensicProps {
  label?: string;
  sublabel?: string;
  className?: string;
}

export const LoadingForensic: React.FC<LoadingForensicProps> = ({
  label = "SYNTHESIZING FORENSIC DATA",
  sublabel = "VERIFYING EVIDENCE TRACEABILITY & GCIT CONDITIONING",
  className,
}) => (
  <div
    className={cn(
      "flex flex-col items-center justify-center p-12 space-y-5 bg-white rounded-md border border-[#DCE4EE]",
      className
    )}
  >
    <div className="relative w-48 h-1.5 bg-[#EEF3F8] overflow-hidden rounded-full border border-[#DCE4EE]">
      <motion.div
        animate={{ x: ["-100%", "100%"] }}
        transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
        className="w-1/2 h-full bg-[#27B3A3] shadow-[0_0_10px_rgba(39,179,163,0.6)] rounded-full"
      />
    </div>
    <div className="flex flex-col items-center space-y-1 text-center">
      <span className="font-mono text-[11px] font-semibold text-[#18263A] tracking-[0.18em] uppercase">
        {label}
      </span>
      <span className="font-mono text-[10px] text-[#7893B3] tracking-wider uppercase">
        {sublabel}
      </span>
    </div>
  </div>
);

export const LoadingSpinner: React.FC<{
  size?: "sm" | "md" | "lg";
  label?: string;
  className?: string;
}> = ({ size = "md", label, className }) => {
  const sizes = {
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-7 h-7",
  };
  return (
    <div className={cn("inline-flex items-center gap-2 text-[#27B3A3]", className)}>
      <Loader2 className={cn("animate-spin", sizes[size])} />
      {label && (
        <span className="font-mono text-xs text-[#5A718E] uppercase tracking-wider">
          {label}
        </span>
      )}
    </div>
  );
};

export const SkeletonBlock: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  ...props
}) => (
  <div
    className={cn(
      "animate-pulse rounded-md bg-[#EEF3F8] border border-[#DCE4EE]/60",
      className
    )}
    {...props}
  />
);

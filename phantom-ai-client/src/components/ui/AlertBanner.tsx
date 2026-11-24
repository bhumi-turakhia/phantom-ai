import React from "react";
import { AlertTriangle, CheckCircle2, Info, ShieldAlert, X } from "lucide-react";
import { cn } from "../../lib/utils";

export interface AlertBannerProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "info" | "warning" | "error" | "success";
  title?: string;
  code?: string;
  onDismiss?: () => void;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({
  variant = "info",
  title,
  code,
  onDismiss,
  className,
  children,
  ...props
}) => {
  const styles = {
    info: {
      box: "bg-[#F4FBFA] border-[#27B3A3]/45 text-[#18263A]",
      iconColor: "text-[#147A6E]",
      badge: "bg-[#27B3A3]/15 text-[#147A6E]",
      Icon: Info,
    },
    warning: {
      box: "bg-[#FFFBEB] border-[#D9A441]/55 text-[#18263A]",
      iconColor: "text-[#B8841E]",
      badge: "bg-[#D9A441]/20 text-[#966912]",
      Icon: AlertTriangle,
    },
    error: {
      box: "bg-[#FEF2F2] border-[#D64550]/45 text-[#18263A]",
      iconColor: "text-[#D64550]",
      badge: "bg-[#D64550]/15 text-[#B4232E]",
      Icon: ShieldAlert,
    },
    success: {
      box: "bg-[#ECFDF5] border-[#1D9E75]/45 text-[#18263A]",
      iconColor: "text-[#1D9E75]",
      badge: "bg-[#1D9E75]/15 text-[#136F52]",
      Icon: CheckCircle2,
    },
  };

  const current = styles[variant];
  const Icon = current.Icon;

  return (
    <div
      role="alert"
      className={cn(
        "relative flex items-start gap-3 px-4 py-3 rounded-md border text-xs",
        current.box,
        className
      )}
      {...props}
    >
      <Icon className={cn("w-4 h-4 shrink-0 mt-0.5", current.iconColor)} />
      <div className="flex-1 min-w-0 space-y-0.5">
        {(title || code) && (
          <div className="flex items-center gap-2">
            {code && (
              <span
                className={cn(
                  "font-mono text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.2 rounded",
                  current.badge
                )}
              >
                {code}
              </span>
            )}
            {title && <span className="font-semibold text-[#18263A]">{title}</span>}
          </div>
        )}
        <div className="text-[#5A718E] leading-relaxed">{children}</div>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss alert"
          className="p-1 rounded text-[#7893B3] hover:text-[#18263A] transition-colors cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
};


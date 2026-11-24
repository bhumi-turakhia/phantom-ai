import React, { useEffect } from "react";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  code?: string;
  size?: "sm" | "md" | "lg" | "xl" | "full";
  footer?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Modal: React.FC<ModalProps> = ({
  open,
  onClose,
  title,
  subtitle,
  code,
  size = "lg",
  footer,
  children,
  className,
}) => {
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const sizeClasses = {
    sm: "max-w-md",
    md: "max-w-xl",
    lg: "max-w-3xl",
    xl: "max-w-5xl",
    full: "max-w-[94vw] h-[90vh]",
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#101B2D]/70 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
      aria-labelledby="forensic-modal-title"
    >
      <div
        className={cn(
          "w-full bg-white rounded-lg border border-[#C1CFDF] shadow-[0_20px_25px_-5px_rgba(16,27,45,0.22)] flex flex-col max-h-[90vh] overflow-hidden",
          sizeClasses[size],
          className
        )}
      >
        {/* Deep Slate Modal Header */}
        <div className="px-6 py-4 bg-[#18263A] text-white border-b border-[#2A3B54] flex items-center justify-between gap-4 shrink-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              {code && (
                <span className="font-mono text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded bg-[#27B3A3]/20 text-[#27B3A3] border border-[#27B3A3]/40">
                  {code}
                </span>
              )}
              <h2
                id="forensic-modal-title"
                className="text-base font-semibold text-white truncate"
              >
                {title}
              </h2>
            </div>
            {subtitle && (
              <p className="text-xs text-[#7893B3] mt-0.5 truncate font-mono">
                {subtitle}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="p-1.5 rounded-md text-[#7893B3] hover:text-white hover:bg-[#101B2D]/60 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6 bg-[#F3F6FA] text-[#18263A]">
          {children}
        </div>

        {/* Optional Footer */}
        {footer && (
          <div className="px-6 py-3.5 bg-white border-t border-[#DCE4EE] flex items-center justify-end gap-3 shrink-0">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
};

export interface DrawerProps extends ModalProps {
  side?: "right" | "left";
}

export const Drawer: React.FC<DrawerProps> = ({
  open,
  onClose,
  title,
  subtitle,
  code,
  side = "right",
  footer,
  children,
  className,
}) => {
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex bg-[#101B2D]/60 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
    >
      <div className="flex-1" onClick={onClose} />
      <div
        className={cn(
          "w-full max-w-lg bg-white h-full border-l border-[#C1CFDF] shadow-2xl flex flex-col",
          side === "left" && "order-first border-l-0 border-r",
          className
        )}
      >
        <div className="px-6 py-4 bg-[#18263A] text-white border-b border-[#2A3B54] flex items-center justify-between gap-4 shrink-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              {code && (
                <span className="font-mono text-[10px] font-semibold uppercase tracking-widest px-1.5 py-0.5 rounded bg-[#27B3A3]/20 text-[#27B3A3]">
                  {code}
                </span>
              )}
              <h2 className="text-sm font-semibold text-white truncate">{title}</h2>
            </div>
            {subtitle && (
              <p className="text-xs text-[#7893B3] mt-0.5 font-mono truncate">
                {subtitle}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close drawer"
            className="p-1.5 rounded-md text-[#7893B3] hover:text-white hover:bg-[#101B2D]/60 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 bg-[#F3F6FA] text-[#18263A]">
          {children}
        </div>

        {footer && (
          <div className="px-6 py-3.5 bg-white border-t border-[#DCE4EE] flex items-center justify-end gap-3 shrink-0">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
};


import React from "react";
import { cn } from "../../lib/utils";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glow?: boolean;
  delay?: number;
  tone?: "default" | "inset" | "slate" | "teal" | "amber";
  noPadding?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, glow = false, tone = "default", noPadding = false, children, ...props }, ref) => {
    const toneStyles = {
      default: "bg-white border-[#DCE4EE] text-[#18263A] shadow-[0_1px_3px_0_rgba(16,27,45,0.06),0_1px_2px_-1px_rgba(16,27,45,0.04)]",
      inset: "bg-[#F8FAFC] border-[#DCE4EE] text-[#18263A]",
      slate: "bg-[#18263A] border-[#2A3B54] text-white shadow-[0_4px_12px_-2px_rgba(16,27,45,0.12)]",
      teal: "bg-[#F4FBFA] border-[#27B3A3]/40 text-[#18263A] shadow-[0_1px_3px_0_rgba(39,179,163,0.08)]",
      amber: "bg-[#FFFBEB] border-[#D9A441]/40 text-[#18263A] shadow-[0_1px_3px_0_rgba(217,164,65,0.08)]",
    };

    return (
      <div
        ref={ref}
        className={cn(
          "relative rounded-md border transition-all duration-150 overflow-hidden",
          toneStyles[tone],
          !noPadding && "p-6",
          glow && "hover:border-[#27B3A3]/60 hover:shadow-[0_4px_14px_-2px_rgba(39,179,163,0.14)]",
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);
Card.displayName = "Card";

export const CardHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn("flex flex-col space-y-1 pb-4 mb-4 border-b border-[#DCE4EE]", className)}
    {...props}
  />
);

export const CardTitle = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3
    className={cn("font-display text-lg font-semibold leading-snug tracking-tight text-[#18263A]", className)}
    {...props}
  />
);

export const CardDescription = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={cn("text-xs text-[#5A718E]", className)} {...props} />
);

export const CardContent = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("pt-0", className)} {...props} />
);

export const CardFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn("flex items-center justify-between pt-4 mt-4 border-t border-[#DCE4EE]", className)}
    {...props}
  />
);

export interface ForensicPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  subtitle?: string;
  code?: string;
  actions?: React.ReactNode;
  footer?: React.ReactNode;
  noPadding?: boolean;
}

export const ForensicPanel: React.FC<ForensicPanelProps> = ({
  title,
  subtitle,
  code,
  actions,
  footer,
  noPadding = false,
  className,
  children,
  ...props
}) => (
  <div
    className={cn(
      "bg-white rounded-md border border-[#DCE4EE] shadow-[0_1px_3px_0_rgba(16,27,45,0.06)] overflow-hidden flex flex-col",
      className
    )}
    {...props}
  >
    <div className="px-5 py-3.5 bg-[#F8FAFC] border-b border-[#DCE4EE] flex items-center justify-between gap-4">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          {code && (
            <span className="font-mono text-[10px] font-semibold uppercase tracking-widest px-1.5 py-0.5 rounded bg-[#18263A] text-[#27B3A3]">
              {code}
            </span>
          )}
          <h3 className="text-sm font-semibold text-[#18263A] truncate">{title}</h3>
        </div>
        {subtitle && <p className="text-xs text-[#5A718E] mt-0.5 truncate">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
    <div className={cn("flex-1", !noPadding && "p-5")}>{children}</div>
    {footer && (
      <div className="px-5 py-3 bg-[#F8FAFC] border-t border-[#DCE4EE] flex items-center justify-between">
        {footer}
      </div>
    )}
  </div>
);

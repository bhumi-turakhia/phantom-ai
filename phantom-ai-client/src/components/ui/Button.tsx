import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "../../lib/utils";

export type ButtonVariant =
  | "primary"
  | "secondary"
  | "slate"
  | "amber"
  | "ghost"
  | "danger";

export type ButtonSize = "xs" | "sm" | "md" | "lg";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

const variants: Record<ButtonVariant, string> = {
  primary:
    "bg-[#27B3A3] text-white hover:bg-[#1F9A8C] active:bg-[#178074] border border-[#1F9A8C] shadow-xs",
  secondary:
    "bg-white text-[#18263A] border border-[#C1CFDF] hover:bg-[#F3F6FA] hover:border-[#7893B3] shadow-2xs",
  slate:
    "bg-[#18263A] text-white hover:bg-[#101B2D] border border-[#101B2D] shadow-xs",
  amber:
    "bg-[#FDF7EB] text-[#9A6B16] border border-[#D9A441]/50 hover:bg-[#F9EBD0]",
  ghost:
    "bg-transparent text-[#3B4F6B] hover:text-[#18263A] hover:bg-[#EEF3F8] border border-transparent",
  danger:
    "bg-white text-[#D64550] border border-[#D64550]/40 hover:bg-[#FDECEC] hover:border-[#D64550]",
};

const sizes: Record<ButtonSize, string> = {
  xs: "h-7 px-2.5 text-xs",
  sm: "h-8 px-3 text-xs",
  md: "h-9 px-4 text-sm",
  lg: "h-11 px-5 text-sm",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "primary",
      size = "md",
      loading,
      leftIcon,
      rightIcon,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-md font-medium tracking-normal cursor-pointer",
          "transition-colors duration-150 disabled:opacity-45 disabled:cursor-not-allowed",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-[#27B3A3]/50",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {loading ? (
          <span className="h-3.5 w-3.5 rounded-full border-2 border-current border-t-transparent animate-spin" />
        ) : (
          leftIcon
        )}
        {children}
        {!loading && rightIcon}
      </button>
    );
  }
);
Button.displayName = "Button";

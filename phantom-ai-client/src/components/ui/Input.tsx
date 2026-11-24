import {
  forwardRef,
  useState,
  type InputHTMLAttributes,
  type TextareaHTMLAttributes,
  type ReactNode,
} from "react";
import { cn } from "../../lib/utils";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
  mono?: boolean;
  leftIcon?: ReactNode;
  rightElement?: ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      label,
      hint,
      error,
      mono = false,
      leftIcon,
      rightElement,
      id,
      ...props
    },
    ref
  ) => {
    return (
      <div className="space-y-1.5">
        {label && (
          <label
            htmlFor={id}
            className="block label-technical text-[#3B4F6B]"
          >
            {label}
          </label>
        )}
        <div className="relative flex items-center">
          {leftIcon && (
            <span className="pointer-events-none absolute left-3 text-[#7893B3]">
              {leftIcon}
            </span>
          )}
          <input
            ref={ref}
            id={id}
            className={cn(
              "w-full h-10 rounded-md bg-white border border-[#C1CFDF] px-3 text-sm text-[#18263A]",
              "placeholder:text-[#7893B3] outline-none transition-colors shadow-2xs",
              "focus:border-[#27B3A3] focus:ring-2 focus:ring-[#27B3A3]/15",
              "disabled:bg-[#EEF3F8] disabled:text-[#7893B3] disabled:cursor-not-allowed",
              mono && "font-mono text-xs",
              leftIcon && "pl-9",
              rightElement && "pr-9",
              error && "border-[#D64550] focus:border-[#D64550] focus:ring-[#D64550]/15",
              className
            )}
            {...props}
          />
          {rightElement && (
            <span className="absolute right-2.5 flex items-center text-[#5A718E]">
              {rightElement}
            </span>
          )}
        </div>
        {error ? (
          <p className="text-xs font-medium text-[#D64550]">{error}</p>
        ) : (
          hint && <p className="text-xs text-[#5A718E]">{hint}</p>
        )}
      </div>
    );
  }
);
Input.displayName = "Input";

export interface TextareaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  hint?: string;
  error?: string;
  maxLength?: number;
  /** 0–100 heuristic score the caller computes from statement content */
  qualityScore?: number;
  showMetrics?: boolean;
}

/**
 * Forensic Witness Statement & Case Note Textarea — structured entry area
 * with character counter and detail indicator.
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      className,
      label,
      hint,
      error,
      maxLength = 2000,
      qualityScore,
      showMetrics = true,
      id,
      value,
      onChange,
      ...props
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = useState((value as string) ?? "");
    const length =
      (value !== undefined ? (value as string) : internalValue)?.length ?? 0;

    const quality =
      qualityScore !== undefined
        ? qualityScore
        : Math.min(100, Math.round((length / maxLength) * 140));

    const qualityLabel =
      quality < 30
        ? "Insufficient detail"
        : quality < 65
        ? "Usable detail"
        : "Strong forensic detail";
    const qualityTone =
      quality < 30
        ? "bg-[#D64550]"
        : quality < 65
        ? "bg-[#D9A441]"
        : "bg-[#1D9E75]";

    return (
      <div className="space-y-1.5">
        {label && (
          <label
            htmlFor={id}
            className="block label-technical text-[#3B4F6B]"
          >
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={id}
          maxLength={maxLength}
          value={value}
          onChange={(e) => {
            if (value === undefined) setInternalValue(e.target.value);
            onChange?.(e);
          }}
          className={cn(
            "w-full min-h-[150px] rounded-md bg-white border border-[#C1CFDF] p-3.5 text-sm text-[#18263A] leading-relaxed",
            "placeholder:text-[#7893B3] outline-none transition-colors resize-y shadow-2xs",
            "focus:border-[#27B3A3] focus:ring-2 focus:ring-[#27B3A3]/15",
            error && "border-[#D64550] focus:border-[#D64550]",
            className
          )}
          {...props}
        />
        {showMetrics && (
          <div className="flex items-center justify-between text-xs pt-0.5">
            <div className="flex items-center gap-2">
              <span className={cn("h-2 w-2 rounded-full", qualityTone)} />
              <span className="text-[#5A718E] label-technical">
                {qualityLabel}
              </span>
            </div>
            <span className="label-technical text-[#7893B3]">
              {length} / {maxLength}
            </span>
          </div>
        )}
        {error ? (
          <p className="text-xs font-medium text-[#D64550]">{error}</p>
        ) : (
          hint && <p className="text-xs text-[#5A718E]">{hint}</p>
        )}
      </div>
    );
  }
);
Textarea.displayName = "Textarea";

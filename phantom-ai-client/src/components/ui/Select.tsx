import { forwardRef, type SelectHTMLAttributes } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "../../lib/utils";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  hint?: string;
  error?: string;
  options?: SelectOption[];
  mono?: boolean;
}

/**
 * Standardized forensic dropdown select component with custom chevron indicator.
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      className,
      label,
      hint,
      error,
      options,
      mono = false,
      id,
      children,
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
        <div className="relative">
          <select
            ref={ref}
            id={id}
            className={cn(
              "w-full h-10 appearance-none rounded-md bg-white border border-[#C1CFDF] pl-3 pr-9 text-sm text-[#18263A]",
              "outline-none transition-colors shadow-2xs cursor-pointer",
              "focus:border-[#27B3A3] focus:ring-2 focus:ring-[#27B3A3]/15",
              "disabled:bg-[#EEF3F8] disabled:text-[#7893B3] disabled:cursor-not-allowed",
              mono && "font-mono text-xs",
              error && "border-[#D64550] focus:border-[#D64550]",
              className
            )}
            {...props}
          >
            {options
              ? options.map((opt) => (
                  <option
                    key={opt.value}
                    value={opt.value}
                    disabled={opt.disabled}
                  >
                    {opt.label}
                  </option>
                ))
              : children}
          </select>
          <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#5A718E]" />
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
Select.displayName = "Select";


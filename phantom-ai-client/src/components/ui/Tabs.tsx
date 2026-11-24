import React, { createContext, useContext } from "react";
import { cn } from "../../lib/utils";

interface TabsContextValue {
  value: string;
  onValueChange: (val: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

export interface TabsProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
  onValueChange: (value: string) => void;
}

export const Tabs: React.FC<TabsProps> = ({
  value,
  onValueChange,
  className,
  children,
  ...props
}) => (
  <TabsContext.Provider value={{ value, onValueChange }}>
    <div className={cn("w-full", className)} {...props}>
      {children}
    </div>
  </TabsContext.Provider>
);

export const TabsList: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div
    role="tablist"
    className={cn(
      "inline-flex items-center gap-1 rounded-md bg-[#EEF3F8] p-1 border border-[#DCE4EE]",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export interface TabsTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
  badge?: string | number;
  icon?: React.ReactNode;
}

export const TabsTrigger: React.FC<TabsTriggerProps> = ({
  value,
  badge,
  icon,
  className,
  children,
  ...props
}) => {
  const ctx = useContext(TabsContext);
  const isActive = ctx?.value === value;

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      onClick={() => ctx?.onValueChange(value)}
      className={cn(
        "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-[4px] font-mono text-xs font-medium transition-all cursor-pointer",
        isActive
          ? "bg-white text-[#18263A] shadow-xs border border-[#DCE4EE]"
          : "text-[#5A718E] hover:text-[#18263A] hover:bg-white/50",
        className
      )}
      {...props}
    >
      {isActive && <span className="w-1.5 h-1.5 rounded-full bg-[#27B3A3] shrink-0" />}
      {icon}
      <span>{children}</span>
      {badge !== undefined && (
        <span
          className={cn(
            "px-1.5 py-0.2 rounded font-mono text-[10px] font-semibold",
            isActive
              ? "bg-[#27B3A3]/15 text-[#147A6E]"
              : "bg-[#DCE4EE] text-[#5A718E]"
          )}
        >
          {badge}
        </span>
      )}
    </button>
  );
};

export interface TabsContentProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
}

export const TabsContent: React.FC<TabsContentProps> = ({
  value,
  className,
  children,
  ...props
}) => {
  const ctx = useContext(TabsContext);
  if (ctx?.value !== value) return null;

  return (
    <div
      role="tabpanel"
      className={cn("mt-4 focus:outline-none", className)}
      {...props}
    >
      {children}
    </div>
  );
};


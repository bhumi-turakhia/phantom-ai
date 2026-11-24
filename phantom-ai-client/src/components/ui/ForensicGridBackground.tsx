import React from "react";
import { cn } from "../../lib/utils";

export type ForensicGridVariant =
  | "workspace"
  | "workspace-subtle"
  | "studio"
  | "auth"
  | "sidebar-static";

export interface ForensicGridBackgroundProps {
  /**
   * Page-specific visibility & tone configuration:
   * - "workspace": Dashboard (standard light blueprint + subtle Forensic Teal ambient illumination)
   * - "workspace-subtle": Cases, Reports, Evidence & Database (extra-restrained light background grid)
   * - "studio": Composite Generation (precision technical grid behind studio panels; never overlays face images)
   * - "auth": Login & Registration (slightly more visible animated blueprint grid for platform entrance)
   * - "sidebar-static": Application Sidebar (Midnight Navy #101B2D static Steel Blue #7893B3 grid)
   */
  variant?: ForensicGridVariant;
  /**
   * Whether the grid moves continuously (default: true for all variants except "sidebar-static").
   * Automatically respects prefers-reduced-motion: reduce via CSS.
   */
  animated?: boolean;
  className?: string;
}

const VARIANT_CLASS_MAP: Record<ForensicGridVariant, string> = {
  workspace: "forensic-grid-workspace",
  "workspace-subtle": "forensic-grid-workspace-subtle",
  studio: "forensic-grid-studio",
  auth: "forensic-grid-auth",
  "sidebar-static": "forensic-grid-sidebar-static",
};

export const ForensicGridBackground: React.FC<ForensicGridBackgroundProps> = ({
  variant = "workspace",
  animated,
  className,
}) => {
  const isAnimated =
    animated !== undefined ? animated : variant !== "sidebar-static";

  return (
    <div
      aria-hidden="true"
      data-testid={`forensic-grid-${variant}`}
      data-grid-variant={variant}
      data-grid-animated={isAnimated ? "true" : "false"}
      className={cn(
        "pointer-events-none select-none absolute inset-0 z-0 overflow-hidden",
        VARIANT_CLASS_MAP[variant],
        className
      )}
    >
      {/* Layer 1: Seamless Technical Blueprint Grid (32px Steel Blue #7893B3 + 160px Forensic Teal #27B3A3) */}
      <div
        className={cn(
          "pointer-events-none absolute inset-0 forensic-grid-pattern",
          isAnimated && "forensic-grid-animated"
        )}
      />

      {/* Layer 2: Restrained Radial Illumination (Soft Steel Blue & Forensic Teal; zero Evidence Amber) */}
      <div className="pointer-events-none absolute inset-0 bg-vignette" />
    </div>
  );
};


/**
 * PHANTOM AI 3.0 — PHASE 9.0 DESIGN SYSTEM TOKENS
 * Option C: Hybrid Forensic Workspace
 *
 * Centralized design tokens for colors, typography, spacing, border radii,
 * shadows, and forensic epistemic evidence states.
 */

export const FORENSIC_COLORS = {
  // Primary Option C — Hybrid Forensic Workspace Core Tokens
  MIDNIGHT_NAVY: "#101B2D",
  DEEP_SLATE: "#18263A",
  FORENSIC_TEAL: "#27B3A3",
  STEEL_BLUE: "#7893B3",
  EVIDENCE_AMBER: "#D9A441",
  MAIN_BACKGROUND: "#F3F6FA",
  PANEL_WHITE: "#FFFFFF",
  PRIMARY_TEXT: "#18263A",

  // Extended Interactive & Surface Palette
  TEAL_HOVER: "#1F9A8C",
  TEAL_ACTIVE: "#178074",
  TEAL_SUBTLE: "#E8F7F5",
  SLATE_ELEVATED: "#21324B",
  SLATE_BORDER: "#293E5C",
  SURFACE_INSET: "#EEF3F8",
  SURFACE_SUBTLE: "#F8FAFC",
  BORDER_DEFAULT: "#DCE4EE",
  BORDER_STRONG: "#C1CFDF",

  // Text Hierarchy on Light Workspace Surfaces
  TEXT_PRIMARY: "#18263A",
  TEXT_SECONDARY: "#3B4F6B",
  TEXT_TERTIARY: "#5A718E",
  TEXT_MUTED: "#7893B3",

  // Text Hierarchy on Dark Navy / Deep Slate Chrome Surfaces
  CHROME_TEXT_PRIMARY: "#F3F6FA",
  CHROME_TEXT_SECONDARY: "#B8C7DA",
  CHROME_TEXT_MUTED: "#7893B3",

  // Semantic & Epistemic Evidence Status Palette
  STATUS_VERIFIED: "#1D9E75",
  STATUS_ACTIVE: "#27B3A3",
  STATUS_PENDING: "#D9A441",
  STATUS_CRITICAL: "#D64550",
  STATUS_CONTRA: "#7C3AED",
} as const;

export const FORENSIC_TYPOGRAPHY = {
  fontFamily: {
    sans: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    mono: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
  },
  fontSize: {
    xs: "0.75rem",     // 12px — dense tables, secondary metadata
    sm: "0.8125rem",   // 13px — standard forensic UI controls & body
    base: "0.875rem",  // 14px — primary workspace text
    md: "1rem",        // 16px — section headers & panel titles
    lg: "1.125rem",    // 18px — page titles
    xl: "1.375rem",    // 22px — key metric readouts
  },
} as const;

export const FORENSIC_SPACING = {
  xs: "0.25rem",   // 4px
  sm: "0.5rem",    // 8px
  md: "0.75rem",   // 12px
  lg: "1rem",      // 16px
  xl: "1.25rem",   // 20px
  "2xl": "1.5rem", // 24px
} as const;

export const FORENSIC_RADII = {
  xs: "2px",
  sm: "4px",
  md: "6px",
  lg: "8px",
} as const;

export const FORENSIC_SHADOWS = {
  panel: "0 1px 3px 0 rgba(16, 27, 45, 0.06), 0 1px 2px -1px rgba(16, 27, 45, 0.04)",
  elevated: "0 4px 12px -2px rgba(16, 27, 45, 0.08), 0 2px 4px -2px rgba(16, 27, 45, 0.04)",
  modal: "0 20px 25px -5px rgba(16, 27, 45, 0.18), 0 8px 10px -6px rgba(16, 27, 45, 0.1)",
  tealFocus: "0 0 0 3px rgba(39, 179, 163, 0.18)",
} as const;

/**
 * Stage 1.5 — Animated Forensic Blueprint Grid Tokens
 * Strictly derived from Midnight Navy (#101B2D), Deep Slate (#18263A),
 * Forensic Teal (#27B3A3), Steel Blue (#7893B3), and Main Background (#F3F6FA).
 * Evidence Amber (#D9A441) is strictly excluded from decorative grid layers.
 */
export const FORENSIC_GRID_TOKENS = {
  cellSizePx: 32,
  majorCellSizePx: 160,
  animationDurationWorkspace: "48s",
  animationDurationAuth: "42s",
  variants: {
    workspace: {
      fineLine: "rgba(120, 147, 179, 0.095)",   // Steel Blue #7893B3
      majorLine: "rgba(39, 179, 163, 0.08)",    // Forensic Teal #27B3A3
      ambientGlow: "rgba(39, 179, 163, 0.045)", // Restrained Teal illumination
    },
    "workspace-subtle": {
      fineLine: "rgba(120, 147, 179, 0.07)",    // Steel Blue #7893B3 (Cases, Reports, Evidence)
      majorLine: "rgba(39, 179, 163, 0.055)",   // Forensic Teal #27B3A3
      ambientGlow: "rgba(120, 147, 179, 0.035)",
    },
    studio: {
      fineLine: "rgba(120, 147, 179, 0.08)",    // Steel Blue #7893B3 (Composite Generation)
      majorLine: "rgba(39, 179, 163, 0.065)",   // Forensic Teal #27B3A3
      ambientGlow: "rgba(39, 179, 163, 0.035)",
    },
    auth: {
      fineLine: "rgba(120, 147, 179, 0.14)",    // Steel Blue #7893B3 (Login & Registration)
      majorLine: "rgba(39, 179, 163, 0.12)",    // Forensic Teal #27B3A3
      ambientGlow: "rgba(39, 179, 163, 0.075)",
    },
    "sidebar-static": {
      fineLine: "rgba(120, 147, 179, 0.065)",   // Static Steel Blue #7893B3 on Midnight Navy #101B2D
      majorLine: "rgba(39, 179, 163, 0.05)",    // Subtle Forensic Teal #27B3A3 subdivision
      ambientGlow: "rgba(39, 179, 163, 0.04)",
    },
  },
} as const;

export type EvidenceEpistemicState =
  | "explicit"
  | "present"
  | "uncertain"
  | "negated"
  | "absent"
  | "contradictory"
  | "unknown";


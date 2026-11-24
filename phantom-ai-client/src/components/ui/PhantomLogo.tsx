import React from "react";
import { cn } from "../../lib/utils";

export interface PhantomLogoProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "full" | "symbol" | "wordmark";
  theme?: "dark" | "light";
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  showTagline?: boolean;
}

const SIZE_CONFIG = {
  xs: { symbol: 20, wordmark: "text-xs", tagline: "text-[8px]" },
  sm: { symbol: 26, wordmark: "text-sm", tagline: "text-[9px]" },
  md: { symbol: 34, wordmark: "text-base", tagline: "text-[10px]" },
  lg: { symbol: 44, wordmark: "text-xl", tagline: "text-[11px]" },
  xl: { symbol: 60, wordmark: "text-2xl sm:text-3xl", tagline: "text-xs" },
};

/**
 * Phantom AI Custom Forensic Symbol:
 * - Stylized Human Facial Reconstruction contour
 * - AI Geometric Delaunay Facial Topology Mesh
 * - Forensic Teal (#27B3A3) Landmark Evidence Nodes
 * - Steel Blue (#7893B3) Viewfinder Focus Brackets with Corner Ticks
 */
export const PhantomSymbol: React.FC<{
  size?: number;
  theme?: "dark" | "light";
  className?: string;
}> = ({ size = 32, theme = "dark", className }) => {
  const isLight = theme === "light";
  const contourColor = isLight ? "#18263A" : "#27B3A3";
  const meshOpacity = isLight ? 0.75 : 0.65;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("shrink-0 transition-transform duration-200 select-none", className)}
      aria-hidden="true"
    >
      {/* 4 Forensic Viewfinder Focus Brackets */}
      <path
        d="M 4 15 L 4 4 L 15 4"
        stroke="#7893B3"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M 49 4 L 60 4 L 60 15"
        stroke="#7893B3"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M 4 49 L 4 60 L 15 60"
        stroke="#7893B3"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M 49 60 L 60 60 L 60 49"
        stroke="#7893B3"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Viewfinder Corner Precision Registration Marks */}
      <circle cx="4" cy="4" r="1.2" fill="#27B3A3" />
      <circle cx="60" cy="4" r="1.2" fill="#27B3A3" />
      <circle cx="4" cy="60" r="1.2" fill="#27B3A3" />
      <circle cx="60" cy="60" r="1.2" fill="#27B3A3" />

      {/* AI Geometric Facial Reconstruction Mesh */}
      <g
        stroke="#7893B3"
        strokeWidth="1"
        strokeOpacity={meshOpacity}
        strokeLinejoin="round"
        strokeLinecap="round"
      >
        {/* Cranial & Forehead Triangulation */}
        <line x1="32" y1="12" x2="32" y2="22" />
        <line x1="22" y1="16" x2="32" y2="22" />
        <line x1="42" y1="16" x2="32" y2="22" />
        <line x1="16" y1="24" x2="24" y2="26" />
        <line x1="24" y1="26" x2="32" y2="22" />
        <line x1="48" y1="24" x2="40" y2="26" />
        <line x1="40" y1="26" x2="32" y2="22" />
        <line x1="22" y1="16" x2="16" y2="24" />
        <line x1="42" y1="16" x2="48" y2="24" />

        {/* Orbit, Nasion & Zygomatic Intersections */}
        <line x1="24" y1="26" x2="32" y2="30" />
        <line x1="40" y1="26" x2="32" y2="30" />
        <line x1="16" y1="24" x2="18" y2="34" />
        <line x1="18" y1="34" x2="24" y2="26" />
        <line x1="48" y1="24" x2="46" y2="34" />
        <line x1="46" y1="34" x2="40" y2="26" />

        {/* Nasal Bridge to Subnasale */}
        <line x1="24" y1="26" x2="27" y2="37" />
        <line x1="40" y1="26" x2="37" y2="37" />
        <line x1="32" y1="30" x2="32" y2="38" />
        <line x1="27" y1="37" x2="32" y2="38" />
        <line x1="37" y1="37" x2="32" y2="38" />
        <line x1="18" y1="34" x2="27" y2="37" />
        <line x1="46" y1="34" x2="37" y2="37" />

        {/* Oral Cavity & Mandibular Angles */}
        <line x1="32" y1="38" x2="32" y2="43" />
        <line x1="27" y1="37" x2="26" y2="44" />
        <line x1="37" y1="37" x2="38" y2="44" />
        <line x1="26" y1="44" x2="32" y2="43" />
        <line x1="38" y1="44" x2="32" y2="43" />
        <line x1="26" y1="44" x2="32" y2="47" />
        <line x1="38" y1="44" x2="32" y2="47" />
        <line x1="18" y1="34" x2="22" y2="46" />
        <line x1="22" y1="46" x2="26" y2="44" />
        <line x1="46" y1="34" x2="42" y2="46" />
        <line x1="42" y1="46" x2="38" y2="44" />

        {/* Chin Apex (Menton) Convergences */}
        <line x1="26" y1="44" x2="32" y2="54" />
        <line x1="38" y1="44" x2="32" y2="54" />
        <line x1="32" y1="47" x2="32" y2="54" />
        <line x1="22" y1="46" x2="32" y2="54" />
        <line x1="42" y1="46" x2="32" y2="54" />
      </g>

      {/* Biometric Axis Highlight Lines */}
      <g stroke="#27B3A3" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
        <line x1="24" y1="26" x2="40" y2="26" />
        <line x1="32" y1="22" x2="32" y2="30" />
        <path d="M 27 37 L 32 38 L 37 37" fill="none" />
        <path d="M 26 44 L 32 43 L 38 44" fill="none" />
      </g>

      {/* Human Facial Silhouette Outline */}
      <path
        d="M 32 12 L 22 16 L 16 24 L 18 34 L 22 46 L 32 54 L 42 46 L 46 34 L 48 24 L 42 16 Z"
        stroke={contourColor}
        strokeWidth="1.6"
        strokeLinejoin="round"
        fill="none"
      />

      {/* Forensic Evidence Landmark Nodes (Teal Precision Circles) */}
      <g fill="#27B3A3">
        <circle cx="32" cy="22" r="1.6" />
        <circle cx="24" cy="26" r="1.9" />
        <circle cx="40" cy="26" r="1.9" />
        <circle cx="18" cy="34" r="1.5" />
        <circle cx="46" cy="34" r="1.5" />
        <circle cx="32" cy="38" r="1.9" />
        <circle cx="26" cy="44" r="1.4" />
        <circle cx="38" cy="44" r="1.4" />
        <circle cx="32" cy="54" r="1.8" />
      </g>
    </svg>
  );
};

export const PhantomLogo: React.FC<PhantomLogoProps> = ({
  variant = "full",
  theme = "dark",
  size = "md",
  showTagline = true,
  className,
  ...props
}) => {
  const config = SIZE_CONFIG[size];
  const isLight = theme === "light";

  if (variant === "symbol") {
    return (
      <div className={cn("inline-flex items-center", className)} {...props}>
        <PhantomSymbol size={config.symbol} theme={theme} />
      </div>
    );
  }

  return (
    <div className={cn("inline-flex items-center gap-3 select-none", className)} {...props}>
      {variant !== "wordmark" && (
        <PhantomSymbol size={config.symbol} theme={theme} />
      )}

      <div className="flex flex-col justify-center min-w-0">
        <div className="flex items-baseline gap-1.5 leading-none">
          <span
            className={cn(
              "font-bold tracking-[0.14em]",
              config.wordmark,
              isLight ? "text-[#18263A]" : "text-white"
            )}
          >
            PHANTOM
          </span>
          <span
            className={cn(
              "font-bold tracking-[0.14em] text-[#27B3A3]",
              config.wordmark
            )}
          >
            AI
          </span>
        </div>

        {showTagline && (
          <span
            className={cn(
              "font-mono font-semibold tracking-[0.18em] uppercase mt-1",
              config.tagline,
              isLight ? "text-[#5A718E]" : "text-[#7893B3]"
            )}
          >
            Forensic Intelligence
          </span>
        )}
      </div>
    </div>
  );
};

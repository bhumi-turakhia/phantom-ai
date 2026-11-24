import { cn } from "../../lib/utils";

/**
 * The recurring visual signature of PHANTOM AI: a set of concentric,
 * uneven ridge-arcs — half fingerprint whorl, half topographic case map.
 * Rendered at low opacity using Steel Blue (#7893B3) by default.
 */
export function FingerprintMotif({
  className,
  opacity = 0.06,
  strokeColor = "#7893B3",
}: {
  className?: string;
  opacity?: number;
  strokeColor?: string;
}) {
  return (
    <svg
      viewBox="0 0 240 240"
      className={cn("pointer-events-none", className)}
      style={{ opacity }}
      aria-hidden="true"
    >
      <g fill="none" stroke={strokeColor} strokeWidth="1">
        <path d="M120 14 C 60 14, 14 60, 14 120 C 14 180, 60 224, 118 226" />
        <path d="M120 30 C 70 30, 30 70, 30 120 C 30 168, 66 206, 114 210" />
        <path d="M120 46 C 80 46, 46 80, 46 120 C 46 158, 74 188, 110 194" />
        <path d="M120 62 C 90 62, 62 90, 62 120 C 62 148, 82 172, 106 178" />
        <path d="M120 78 C 100 78, 78 100, 78 120 C 78 138, 92 156, 102 162" />
        <path d="M132 14 C 168 18, 200 50, 206 86" />
        <path d="M138 30 C 164 34, 188 58, 192 84" />
        <path d="M144 46 C 162 50, 178 64, 180 82" />
      </g>
    </svg>
  );
}

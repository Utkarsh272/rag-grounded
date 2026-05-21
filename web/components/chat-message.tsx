// web/components/chat-message.tsx
"use client";

import { useState } from "react";
import type { Citation, ClaimScore } from "@/lib/api";

interface Props {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  claimScores?: ClaimScore[];
  abstained?: boolean;
  isStreaming?: boolean;
  onCitationClick?: (citation: Citation) => void;
  activeCitationId?: number | null;
}

// Parse answer text into segments: plain text and [N] citation markers
function parseContent(content: string): Array<{ type: "text" | "cite"; value: string; num?: number }> {
  const parts: Array<{ type: "text" | "cite"; value: string; num?: number }> = [];
  const regex = /\[(\d+)\]/g;
  let last = 0;
  let match;

  while ((match = regex.exec(content)) !== null) {
    if (match.index > last) {
      parts.push({ type: "text", value: content.slice(last, match.index) });
    }
    parts.push({ type: "cite", value: match[0], num: parseInt(match[1]) });
    last = match.index + match[0].length;
  }

  if (last < content.length) {
    parts.push({ type: "text", value: content.slice(last) });
  }

  return parts;
}

export function ChatMessage({
  role,
  content,
  citations = [],
  claimScores = [],
  abstained = false,
  isStreaming = false,
  onCitationClick,
  activeCitationId,
}: Props) {
  const [showScores, setShowScores] = useState(false);
  const hasLowConfidence = claimScores.some((c) => c.low_confidence);
  const segments = role === "assistant" ? parseContent(content) : null;

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-tr-sm bg-indigo-600 text-white text-sm leading-relaxed">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 max-w-full">
      {/* Avatar */}
      <div className="shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center mt-0.5">
        <svg className="w-3.5 h-3.5 text-white" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
      </div>

      <div className="flex-1 min-w-0 space-y-2">
        {/* Abstain banner */}
        {abstained && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-400">
            <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z" />
            </svg>
            Insufficient information to answer confidently
          </div>
        )}

        {/* Answer text with inline citation chips */}
        <div className="text-sm text-zinc-200 leading-relaxed">
          {segments?.map((seg, i) => {
            if (seg.type === "text") {
              return <span key={i}>{seg.value}</span>;
            }
            // Citation chip
            const citation = citations.find((c) => c.id === seg.num);
            const isActive = activeCitationId === seg.num;
            return (
              <button
                key={i}
                onClick={() => citation && onCitationClick?.(citation)}
                className={`
                  inline-flex items-center justify-center
                  w-5 h-5 mx-0.5 rounded text-[10px] font-bold
                  transition-all duration-150 align-middle
                  ${isActive
                    ? "bg-indigo-500 text-white"
                    : "bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/40 hover:text-indigo-200"
                  }
                `}
              >
                {seg.num}
              </button>
            );
          })}

          {/* Streaming cursor */}
          {isStreaming && (
            <span className="inline-block w-0.5 h-4 bg-indigo-400 ml-0.5 align-middle animate-pulse" />
          )}
        </div>

        {/* Confidence scores row */}
        {claimScores.length > 0 && (
          <div>
            <button
              onClick={() => setShowScores((s) => !s)}
              className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
            >
              {hasLowConfidence ? (
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              ) : (
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              )}
              {hasLowConfidence ? "Some claims have low confidence" : "All claims verified"}
              <svg
                className={`w-3 h-3 transition-transform ${showScores ? "rotate-180" : ""}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {showScores && (
              <div className="mt-2 space-y-1.5 pl-3 border-l border-zinc-800">
                {claimScores.map((cs, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <div className="mt-1.5 shrink-0">
                      {cs.low_confidence ? (
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 block" />
                      ) : (
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 block" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-zinc-400 leading-snug">{cs.claim}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${
                              cs.low_confidence ? "bg-amber-400" : "bg-emerald-400"
                            }`}
                            style={{ width: `${Math.round(cs.score * 100)}%` }}
                          />
                        </div>
                        <span className="text-[10px] text-zinc-600 tabular-nums w-8 text-right">
                          {Math.round(cs.score * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

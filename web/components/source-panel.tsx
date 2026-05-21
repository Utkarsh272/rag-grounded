// web/components/source-panel.tsx
"use client";

import type { Citation } from "@/lib/api";

interface Props {
  citations: Citation[];
  activeCitationId: number | null;
  onClose: () => void;
}

export function SourcePanel({ citations, activeCitationId, onClose }: Props) {
  const active = citations.find((c) => c.id === activeCitationId);

  if (!active) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <span className="w-5 h-5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-bold flex items-center justify-center">
            {active.id}
          </span>
          <span className="text-xs font-medium text-zinc-300">Source</span>
        </div>
        <button
          onClick={onClose}
          className="w-6 h-6 rounded flex items-center justify-center text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Section label */}
        {active.section_title && (
          <div className="flex items-center gap-2">
            <svg className="w-3.5 h-3.5 text-zinc-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
            </svg>
            <span className="text-xs text-zinc-500 font-medium">{active.section_title}</span>
          </div>
        )}

        {/* Highlighted chunk text */}
        <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-3">
          <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
            {active.text}
            {active.text.length >= 300 && (
              <span className="text-zinc-600"> …</span>
            )}
          </p>
        </div>

        {/* Char offset metadata */}
        <div className="flex gap-4">
          <div className="flex-1 rounded-md bg-zinc-900 border border-zinc-800/50 px-3 py-2">
            <p className="text-[10px] text-zinc-600 uppercase tracking-wider mb-0.5">Start</p>
            <p className="text-xs font-mono text-zinc-400">{active.start_char.toLocaleString()}</p>
          </div>
          <div className="flex-1 rounded-md bg-zinc-900 border border-zinc-800/50 px-3 py-2">
            <p className="text-[10px] text-zinc-600 uppercase tracking-wider mb-0.5">End</p>
            <p className="text-xs font-mono text-zinc-400">{active.end_char.toLocaleString()}</p>
          </div>
          <div className="flex-1 rounded-md bg-zinc-900 border border-zinc-800/50 px-3 py-2">
            <p className="text-[10px] text-zinc-600 uppercase tracking-wider mb-0.5">Length</p>
            <p className="text-xs font-mono text-zinc-400">
              {(active.end_char - active.start_char).toLocaleString()}
            </p>
          </div>
        </div>

        {/* All citations in this conversation */}
        {citations.length > 1 && (
          <div>
            <p className="text-[10px] text-zinc-600 uppercase tracking-wider mb-2">All sources</p>
            <div className="space-y-1">
              {citations.map((c) => (
                <div
                  key={c.id}
                  className={`
                    flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs cursor-default
                    transition-colors
                    ${c.id === activeCitationId
                      ? "bg-indigo-500/15 border border-indigo-500/25"
                      : "bg-zinc-900/50 border border-transparent"
                    }
                  `}
                >
                  <span className={`
                    w-4 h-4 rounded text-[9px] font-bold flex items-center justify-center shrink-0
                    ${c.id === activeCitationId ? "bg-indigo-500 text-white" : "bg-zinc-800 text-zinc-400"}
                  `}>
                    {c.id}
                  </span>
                  <span className="text-zinc-400 truncate">
                    {c.section_title ?? `chars ${c.start_char}–${c.end_char}`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

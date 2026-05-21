// web/components/document-list.tsx
"use client";

import { useEffect, useState, useCallback } from "react";
import { listDocuments, pollDocumentStatus, type Document } from "@/lib/api";

interface Props {
  selectedId: string | null;
  onSelect: (doc: Document) => void;
  refreshTrigger: number; // increment from parent to force refresh
}

const STATUS_DOT: Record<Document["status"], string> = {
  complete:   "bg-emerald-400",
  processing: "bg-amber-400 animate-pulse",
  pending:    "bg-zinc-600 animate-pulse",
  failed:     "bg-red-400",
};

const STATUS_LABEL: Record<Document["status"], string> = {
  complete:   "Ready",
  processing: "Processing…",
  pending:    "Queued",
  failed:     "Failed",
};

export function DocumentList({ selectedId, onSelect, refreshTrigger }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocs(data);
    } finally {
      setLoading(false);
    }
  }, []);

  // Poll processing documents every 2s until they complete
  useEffect(() => {
    load();
  }, [load, refreshTrigger]);

  useEffect(() => {
    const processing = docs.filter((d) => d.status === "processing" || d.status === "pending");
    if (processing.length === 0) return;

    const timer = setInterval(async () => {
      const updated = await Promise.all(
        processing.map((d) => pollDocumentStatus(d.id))
      );
      setDocs((prev) =>
        prev.map((d) => updated.find((u) => u.id === d.id) ?? d)
      );
    }, 2000);

    return () => clearInterval(timer);
  }, [docs]);

  if (loading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-14 rounded-lg bg-zinc-800/50 animate-pulse" />
        ))}
      </div>
    );
  }

  if (docs.length === 0) {
    return (
      <p className="text-xs text-zinc-600 text-center py-6">
        No documents yet. Upload one above.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {docs.map((doc) => (
        <button
          key={doc.id}
          onClick={() => doc.status === "complete" && onSelect(doc)}
          disabled={doc.status !== "complete"}
          className={`
            w-full text-left px-3 py-3 rounded-lg transition-all duration-150
            flex items-start gap-3 group
            ${selectedId === doc.id
              ? "bg-indigo-600/20 border border-indigo-500/30"
              : "hover:bg-zinc-800/80 border border-transparent"
            }
            ${doc.status !== "complete" ? "opacity-60 cursor-default" : "cursor-pointer"}
          `}
        >
          {/* File type icon */}
          <div className="mt-0.5 w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center shrink-0 text-[10px] font-mono font-bold text-zinc-500 uppercase">
            {doc.source_type}
          </div>

          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-zinc-200 truncate leading-tight">
              {doc.title}
            </p>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOT[doc.status]}`} />
              <span className="text-xs text-zinc-500">{STATUS_LABEL[doc.status]}</span>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

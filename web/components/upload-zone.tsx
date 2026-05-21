// web/components/upload-zone.tsx
"use client";

import { useCallback, useState } from "react";
import { uploadDocument } from "@/lib/api";

interface Props {
  onUploaded: (documentId: string) => void;
}

export function UploadZone({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handle = useCallback(
    async (file: File) => {
      setError(null);
      setUploading(true);
      try {
        const { document_id } = await uploadDocument(file);
        onUploaded(document_id);
      } catch (e) {
        setError(String(e));
      } finally {
        setUploading(false);
      }
    },
    [onUploaded]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handle(file);
    },
    [handle]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`
        relative border-2 border-dashed rounded-xl px-8 py-12 text-center
        transition-all duration-200 cursor-pointer group
        ${dragging
          ? "border-indigo-400 bg-indigo-500/10"
          : "border-zinc-700 hover:border-zinc-500 bg-zinc-900/50 hover:bg-zinc-800/50"
        }
      `}
      onClick={() => document.getElementById("file-input")?.click()}
    >
      <input
        id="file-input"
        type="file"
        accept=".pdf,.md,.txt"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); }}
      />

      {uploading ? (
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-zinc-400">Uploading and processing…</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <div className={`
            w-12 h-12 rounded-xl flex items-center justify-center
            transition-colors duration-200
            ${dragging ? "bg-indigo-500/20" : "bg-zinc-800 group-hover:bg-zinc-700"}
          `}>
            <svg className="w-6 h-6 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-zinc-300">
              Drop a file or <span className="text-indigo-400">click to browse</span>
            </p>
            <p className="text-xs text-zinc-600 mt-1">PDF, Markdown, or plain text</p>
          </div>
        </div>
      )}

      {error && (
        <p className="mt-3 text-xs text-red-400">{error}</p>
      )}
    </div>
  );
}

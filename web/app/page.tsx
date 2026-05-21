// web/app/page.tsx
"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { UploadZone } from "@/components/upload-zone";
import { DocumentList } from "@/components/document-list";
import { ChatMessage } from "@/components/chat-message";
import { SourcePanel } from "@/components/source-panel";
import { askQuestion, createConversation, getConversation } from "@/lib/api";
import type { Citation, ClaimScore, Document, Message } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  claimScores: ClaimScore[];
  abstained: boolean;
  isStreaming?: boolean;
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [activeCitationId, setActiveCitationId] = useState<number | null>(null);
  const [allCitations, setAllCitations] = useState<Citation[]>([]);
  const [refreshDocs, setRefreshDocs] = useState(0);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // When a document is selected, create or load a conversation
  const handleSelectDoc = useCallback(async (doc: Document) => {
    setSelectedDoc(doc);
    setMessages([]);
    setAllCitations([]);
    setActiveCitationId(null);
    setConversationId(null);
    setSending(false);

    try {
      const conv = await createConversation(doc.id);
      setConversationId(conv.id);
      // Load any existing messages
      const full = await getConversation(conv.id);
      if (full.messages && full.messages.length > 0) {
        setMessages(
          full.messages.map((m: Message) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            citations: m.citations ?? [],
            claimScores: m.claim_scores ?? [],
            abstained: m.abstained ?? false,
          }))
        );
      }
    } catch (e) {
      console.error("Failed to create conversation:", e);
    }
  }, []);

  const handleUpload = useCallback((documentId: string) => {
    setRefreshDocs((n) => n + 1);
    // Don't auto-select — user needs to wait for processing
  }, []);

  const handleSend = useCallback(async () => {
    const question = input.trim();
    if (!question || !conversationId || sending) return;

    setInput("");
    setSending(true);
    setActiveCitationId(null);

    // Optimistically add user message
    const userMsgId = `user-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: question, citations: [], claimScores: [], abstained: false },
    ]);

    // Add streaming assistant placeholder
    const asstMsgId = `asst-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: asstMsgId, role: "assistant", content: "", citations: [], claimScores: [], abstained: false, isStreaming: true },
    ]);

    const cancel = askQuestion(conversationId, question, {
      onToken: (text) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId ? { ...m, content: m.content + text } : m
          )
        );
      },
      onCitation: (citation) => {
        setAllCitations((prev) => {
          const exists = prev.find((c) => c.id === citation.id);
          return exists ? prev : [...prev, citation];
        });
      },
      onComplete: (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId
              ? {
                  ...m,
                  id: data.message_id ?? asstMsgId,
                  content: data.answer,
                  citations: data.citations,
                  claimScores: data.claim_scores,
                  abstained: data.abstained,
                  isStreaming: false,
                }
              : m
          )
        );
        setAllCitations(data.citations);
        setSending(false);
      },
      onError: (detail) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId
              ? { ...m, content: `Error: ${detail}`, isStreaming: false }
              : m
          )
        );
        setSending(false);
      },
    });

    cancelRef.current = cancel;
  }, [input, conversationId, sending]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const sourcePanelOpen = activeCitationId !== null;

  return (
    <div className="h-screen flex bg-zinc-950 text-zinc-100 overflow-hidden font-sans">

      {/* ── Sidebar ────────────────────────────────────────────────────────── */}
      <aside className="w-72 shrink-0 flex flex-col border-r border-zinc-800/60 bg-zinc-950">
        {/* Logo */}
        <div className="px-4 py-4 border-b border-zinc-800/60">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-white" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold text-zinc-100 leading-none">RAG Citations</p>
              <p className="text-[10px] text-zinc-500 mt-0.5">Grounded document Q&A</p>
            </div>
          </div>
        </div>

        {/* Upload */}
        <div className="p-3 border-b border-zinc-800/60">
          <UploadZone onUploaded={handleUpload} />
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto p-3">
          <p className="text-[10px] text-zinc-600 uppercase tracking-wider mb-2 px-1">Documents</p>
          <DocumentList
            selectedId={selectedDoc?.id ?? null}
            onSelect={handleSelectDoc}
            refreshTrigger={refreshDocs}
          />
        </div>
      </aside>

      {/* ── Chat area ──────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="px-6 py-3.5 border-b border-zinc-800/60 flex items-center justify-between shrink-0">
          <div>
            {selectedDoc ? (
              <>
                <p className="text-sm font-medium text-zinc-200 leading-none">{selectedDoc.title}</p>
                <p className="text-xs text-zinc-500 mt-0.5">Ask anything about this document</p>
              </>
            ) : (
              <p className="text-sm text-zinc-500">Select a document to start</p>
            )}
          </div>
          {selectedDoc && (
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Ready
            </div>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {!selectedDoc && (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
              <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                <svg className="w-8 h-8 text-zinc-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-400">No document selected</p>
                <p className="text-xs text-zinc-600 mt-1">Upload a PDF or Markdown file, then select it to start chatting</p>
              </div>
            </div>
          )}

          {selectedDoc && messages.length === 0 && !sending && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
              <p className="text-sm text-zinc-500">Ask a question about <span className="text-zinc-300">{selectedDoc.title}</span></p>
              <div className="flex flex-wrap gap-2 justify-center max-w-md">
                {["What is this document about?", "Summarize the key points", "What are the main conclusions?"].map((s) => (
                  <button
                    key={s}
                    onClick={() => { setInput(s); inputRef.current?.focus(); }}
                    className="px-3 py-1.5 text-xs rounded-full border border-zinc-800 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage
              key={msg.id}
              role={msg.role}
              content={msg.content}
              citations={msg.citations}
              claimScores={msg.claimScores}
              abstained={msg.abstained}
              isStreaming={msg.isStreaming}
              activeCitationId={activeCitationId}
              onCitationClick={(c) =>
                setActiveCitationId((prev) => (prev === c.id ? null : c.id))
              }
            />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="px-6 pb-6 pt-2 shrink-0">
          <div className={`
            flex items-end gap-3 rounded-xl border px-4 py-3 transition-colors
            ${selectedDoc
              ? "border-zinc-700 bg-zinc-900/80 focus-within:border-zinc-600"
              : "border-zinc-800 bg-zinc-900/40 opacity-50"
            }
          `}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!selectedDoc || sending}
              placeholder={selectedDoc ? "Ask a question… (Enter to send)" : "Select a document first"}
              rows={1}
              className="flex-1 bg-transparent text-sm text-zinc-200 placeholder:text-zinc-600 resize-none outline-none max-h-32 leading-relaxed"
              style={{ overflowY: "auto" }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || !selectedDoc || sending}
              className={`
                shrink-0 w-8 h-8 rounded-lg flex items-center justify-center
                transition-all duration-150
                ${input.trim() && selectedDoc && !sending
                  ? "bg-indigo-600 hover:bg-indigo-500 text-white"
                  : "bg-zinc-800 text-zinc-600 cursor-default"
                }
              `}
            >
              {sending ? (
                <div className="w-3.5 h-3.5 border border-zinc-500 border-t-zinc-300 rounded-full animate-spin" />
              ) : (
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </main>

      {/* ── Source panel ────────────────────────────────────────────────────── */}
      <aside
        className={`
          shrink-0 border-l border-zinc-800/60 bg-zinc-950 transition-all duration-300 overflow-hidden
          ${sourcePanelOpen ? "w-80" : "w-0"}
        `}
      >
        {sourcePanelOpen && (
          <SourcePanel
            citations={allCitations}
            activeCitationId={activeCitationId}
            onClose={() => setActiveCitationId(null)}
          />
        )}
      </aside>
    </div>
  );
}

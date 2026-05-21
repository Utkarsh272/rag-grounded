// web/lib/api.ts
// All backend API calls. Base URL from env, falls back to localhost for dev.

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Document {
  id: string;
  title: string;
  source_type: string;
  status: "pending" | "processing" | "complete" | "failed";
  error_message?: string;
  created_at: string;
}

export interface Citation {
  id: number;
  chunk_id: string;
  section_title: string | null;
  start_char: number;
  end_char: number;
  text: string;
}

export interface ClaimScore {
  claim: string;
  score: number;
  low_confidence: boolean;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  claim_scores: ClaimScore[] | null;
  abstained: boolean;
  retrieval_meta: Record<string, unknown> | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  document_id: string;
  title: string;
  created_at: string;
  messages?: Message[];
}

// ── Documents ────────────────────────────────────────────────────────────────

export async function uploadDocument(file: File): Promise<{ document_id: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/v1/documents`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listDocuments(): Promise<Document[]> {
  const res = await fetch(`${BASE}/v1/documents`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function pollDocumentStatus(id: string): Promise<Document> {
  const res = await fetch(`${BASE}/v1/documents/${id}/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Conversations ─────────────────────────────────────────────────────────────

export async function createConversation(document_id: string): Promise<Conversation> {
  const res = await fetch(`${BASE}/v1/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getConversation(id: string): Promise<Conversation> {
  const res = await fetch(`${BASE}/v1/conversations/${id}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── SSE message stream ────────────────────────────────────────────────────────

export interface SSEHandlers {
  onToken: (text: string) => void;
  onCitation: (citation: Citation) => void;
  onComplete: (data: {
    message_id: string;
    answer: string;
    citations: Citation[];
    claim_scores: ClaimScore[];
    abstained: boolean;
    retrieval_meta: Record<string, unknown>;
  }) => void;
  onError: (detail: string) => void;
}

export function askQuestion(
  conversationId: string,
  question: string,
  handlers: SSEHandlers
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/v1/conversations/${conversationId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        handlers.onError(`Request failed: ${res.status}`);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE messages are separated by double newlines
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const lines = part.trim().split("\n");
          let event = "message";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) event = line.slice(7);
            if (line.startsWith("data: ")) dataStr = line.slice(6);
          }

          if (!dataStr) continue;
          try {
            const data = JSON.parse(dataStr);
            if (event === "token") handlers.onToken(data.text);
            else if (event === "citation") handlers.onCitation(data);
            else if (event === "complete") handlers.onComplete(data);
            else if (event === "error") handlers.onError(data.detail);
          } catch {
            // malformed SSE chunk — skip
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        handlers.onError(String(err));
      }
    }
  })();

  return () => controller.abort();
}

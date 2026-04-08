'use client';

import { useState, useRef, useEffect, useCallback, DragEvent } from 'react';
import axios from 'axios';
import {
  Send, Upload, FileText, Bot, User, Loader2, CheckCircle,
  Zap, Database, ChevronDown, ChevronUp, RotateCw, Sparkles,
} from 'lucide-react';

// Backend URL — adjust if you change the uvicorn port
const API_URL = 'http://localhost:8000/api';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Message {
  id: number;
  role: 'user' | 'bot';
  content: string;
  sources?: string[];
  isError?: boolean;
}

interface IndexedDoc {
  source: string;       // filename
  chunk_count: number;  // chunks stored in ChromaDB
  cached?: boolean;     // true = was already indexed before this session
}

// ─── Suggested questions shown on empty state ────────────────────────────────
const SUGGESTIONS = [
  'Who is this person?',
  'What are the key skills mentioned?',
  'Summarize this document',
  'What is the work experience?',
];

// ─── Main Component ──────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [docs, setDocs] = useState<IndexedDoc[]>([]);
  const [expandedSources, setExpandedSources] = useState<number | null>(null);

  const idCounter = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── Auto-scroll to bottom ──
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Auto-resize textarea ──
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  }, [input]);

  // ── Load already-indexed docs on mount ──
  useEffect(() => {
    axios.get(`${API_URL}/documents`)
      .then(res => {
        const loaded: IndexedDoc[] = res.data.documents.map(
          (d: { source: string; chunk_count: number }) => ({
            source: d.source,
            chunk_count: d.chunk_count,
            cached: true,   // all pre-existing docs are "cached"
          })
        );
        setDocs(loaded);
      })
      .catch(() => {
        /* backend not running yet — silently ignore */
      });
  }, []);

  // ─── next message id ────
  const nextId = () => {
    idCounter.current += 1;
    return idCounter.current;
  };

  // ── Add a bot message ──
  const addBotMsg = (content: string, sources?: string[], isError = false) => {
    setMessages(prev => [
      ...prev,
      { id: nextId(), role: 'bot', content, sources, isError },
    ]);
  };

  // ──────────────────────────────────────────────────────────────────────────
  // UPLOAD HANDLER
  // Sends file to /api/upload.
  // If the file is already indexed (already_indexed=true), shows a "cached"
  // notice instead of a full processing message.
  // ──────────────────────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file: File, forceReprocess = false) => {
    if (!file.name) return;
    const ALLOWED = ['.pdf', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.webp'];
    const lowerName = file.name.toLowerCase();
    if (!ALLOWED.some(ext => lowerName.endsWith(ext))) {
      addBotMsg('⚠️ Supported files: PDF, TXT, PNG, JPG, JPEG, BMP, WEBP', undefined, true);
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(
        `${API_URL}/upload?force=${forceReprocess}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      const { filename, chunks, already_indexed } = res.data;

      // Update document sidebar
      setDocs(prev => {
        const exists = prev.find(d => d.source === filename);
        if (exists) return prev; // already shown
        return [...prev, { source: filename, chunk_count: chunks, cached: already_indexed }];
      });

      if (already_indexed) {
        addBotMsg(
          `⚡ "${filename}" is already indexed (${chunks} chunks cached). You can ask questions about it right away — no re-upload needed!`
        );
      } else {
        addBotMsg(
          `✅ "${filename}" uploaded and indexed into ${chunks} chunks. Ask me anything about it!`
        );
      }
    } catch {
      addBotMsg('❌ Upload failed. Is the backend running on port 8000?', undefined, true);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, []);

  // ── File input change ──
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  // ── Drag & drop ──
  const onDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUpload(file);
  };

  // ──────────────────────────────────────────────────────────────────────────
  // ASK HANDLER  — sends question to /api/ask (full RAG pipeline)
  // ──────────────────────────────────────────────────────────────────────────
  const handleSend = async (overrideText?: string) => {
    const question = (overrideText ?? input).trim();
    if (!question || isLoading) return;

    setInput('');
    setMessages(prev => [
      ...prev,
      { id: nextId(), role: 'user', content: question },
    ]);
    setIsLoading(true);

    try {
      const res = await axios.post(`${API_URL}/ask`, {
        question,
        top_k: 5,
      });
      addBotMsg(res.data.answer, res.data.sources);
    } catch {
      addBotMsg(
        'Something went wrong. Please make sure the backend is running.',
        undefined,
        true
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Re-process a document in the sidebar ──
  const reprocessDoc = async (source: string) => {
    const fake = new File([''], source);   // fake File to reuse upload handler
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await handleUpload(fake as any, true);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', height: '100vh', position: 'relative', zIndex: 1 }}>

      {/* ── SIDEBAR ─────────────────────────────────────────────── */}
      <aside className="sidebar">

        {/* Logo */}
        <div className="sidebar-header">
          <div className="logo-wrapper">
            <div className="logo-icon">
              <Sparkles size={20} />
            </div>
            <div className="logo-text">
              <h1>RAG Chatbot</h1>
              <p>Intelligent Document Q&A</p>
            </div>
          </div>
        </div>

        {/* Upload zone */}
        <label
          htmlFor="file-input"
          className={`upload-zone ${isDragging ? 'dragging' : ''} ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <input
            type="file"
            id="file-input"
            ref={fileInputRef}
            onChange={onFileChange}
            accept=".pdf,.txt,.png,.jpg,.jpeg,.bmp,.webp"
            className="hidden"
            disabled={isUploading}
          />
          <div className="upload-icon-ring">
            {isUploading
              ? <Loader2 size={20} color="var(--accent-glow)" className="animate-spin" />
              : <Upload size={20} color="var(--accent-glow)" />
            }
          </div>
          <span className="upload-label">
            {isUploading ? 'Processing…' : isDragging ? 'Drop it!' : 'Upload Document'}
          </span>
          <span className="upload-hint">
            {isUploading ? 'Embedding chunks into ChromaDB' : 'PDF · TXT · Image · Drag & drop or click'}
          </span>
          {isUploading && <div className="upload-progress-bar" />}
        </label>

        {/* Indexed documents list */}
        <div className="docs-section">
          <p className="docs-label" style={{ marginBottom: 12 }}>
            <Database size={10} style={{ display: 'inline', marginRight: 5 }} />
            Indexed Documents ({docs.length})
          </p>

          {docs.length === 0 ? (
            <p style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic', padding: '0 4px' }}>
              No documents indexed yet.
              <br />Upload a PDF to get started.
            </p>
          ) : (
            docs.map((doc, i) => (
              <div
                key={doc.source}
                id={`doc-entry-${i}`}
                className={`doc-entry ${doc.cached ? 'already-indexed' : ''}`}
                title={doc.source}
              >
                <div className="doc-icon">
                  <FileText size={14} color="var(--accent-glow)" />
                </div>
                <div className="doc-info">
                  <p className="doc-name">{doc.source}</p>
                  <p className="doc-meta">
                    {doc.chunk_count} chunks
                    {doc.cached ? ' · cached ⚡' : ' · new'}
                  </p>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                  <div className={`status-dot ${doc.cached ? 'cached' : ''}`} />
                  <button
                    onClick={() => reprocessDoc(doc.source)}
                    title="Re-index this document"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--text-muted)', padding: 0, lineHeight: 1,
                    }}
                  >
                    <RotateCw size={11} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="sidebar-footer">
          <div className="powered-badge">
            <Zap size={11} />
            Powered by <span>Groq</span> + <span>ChromaDB</span>
          </div>
        </div>
      </aside>

      {/* ── MAIN CHAT ─────────────────────────────────────────────── */}
      <main className="chat-area">

        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-left">
            <Bot size={18} color="var(--accent-glow)" />
            <div>
              <p className="chat-header-title">Document Assistant</p>
              <p className="chat-header-sub">
                {docs.length > 0
                  ? `${docs.length} document${docs.length > 1 ? 's' : ''} indexed`
                  : 'Upload a document to begin'}
              </p>
            </div>
          </div>
          <div className="online-badge">
            <div className="online-dot" />
            Online
          </div>
        </div>

        {/* Messages */}
        <div className="messages-container">

          {/* ─ Empty state ─ */}
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-orb">
                <Sparkles size={32} color="var(--accent-glow)" />
              </div>
              <h2 className="empty-title">Ask me anything</h2>
              <p className="empty-desc">
                Upload a PDF, TXT, or image (PNG/JPG) using the sidebar, then ask questions.
                I'll answer based entirely on your document — no hallucination.
              </p>
              {docs.length > 0 && (
                <div className="empty-chips">
                  {SUGGESTIONS.map(s => (
                    <button
                      key={s}
                      className="empty-chip"
                      onClick={() => handleSend(s)}
                      id={`suggestion-${s.replace(/\s+/g, '-').toLowerCase()}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ─ Message list ─ */}
          {messages.map(msg => (
            <div key={msg.id} className={`message-row ${msg.role}`} id={`msg-${msg.id}`}>

              {/* Avatar */}
              <div className={`avatar ${msg.role}`}>
                {msg.role === 'bot'
                  ? <Bot size={16} color="white" />
                  : <User size={16} />
                }
              </div>

              {/* Bubble + sources */}
              <div className="bubble">
                <div
                  className={`bubble-text ${msg.isError ? 'error-bubble' : ''}`}
                  style={msg.isError ? { borderColor: 'rgba(244,63,94,0.3)' } : {}}
                >
                  {msg.content}
                </div>

                {/* Sources toggle */}
                {msg.sources && msg.sources.length > 0 && (
                  <div>
                    <button
                      className="sources-toggle"
                      onClick={() => setExpandedSources(
                        expandedSources === msg.id ? null : msg.id
                      )}
                      id={`sources-toggle-${msg.id}`}
                    >
                      <FileText size={11} />
                      {expandedSources === msg.id ? 'Hide' : 'View'} sources
                      &nbsp;({msg.sources.length})
                      {expandedSources === msg.id
                        ? <ChevronUp size={11} />
                        : <ChevronDown size={11} />
                      }
                    </button>

                    {expandedSources === msg.id && (
                      <div className="sources-panel">
                        <p className="sources-panel-header">Context chunks used</p>
                        {msg.sources.map((src, j) => (
                          <div key={j} className="source-chip">{src}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Right-side check for bot (shows ✓ after answer) */}
              {msg.role === 'bot' && !msg.isError && (
                <CheckCircle
                  size={14}
                  color="var(--green)"
                  style={{ marginTop: 10, flexShrink: 0, opacity: 0.7 }}
                />
              )}
            </div>
          ))}

          {/* ─ Typing indicator ─ */}
          {isLoading && (
            <div className="typing-row" id="typing-indicator">
              <div className="avatar bot">
                <Bot size={16} color="white" />
              </div>
              <div className="typing-bubble">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* ── Input area ─────────────────────────────────────────── */}
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="chat-textarea"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                docs.length > 0
                  ? 'Ask anything about your documents… (Enter to send)'
                  : 'Upload a document first, then ask questions…'
              }
              rows={1}
            />
            <button
              id="send-btn"
              className="send-btn"
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
            >
              {isLoading
                ? <Loader2 size={16} color="white" className="animate-spin" />
                : <Send size={16} color="white" />
              }
            </button>
          </div>
          <p className="input-hint">
            Shift+Enter for new line · Answers generated from your documents via RAG
          </p>
        </div>
      </main>
    </div>
  );
}

import type { FormEvent, RefObject } from 'react';
import type { Conversation, GroupedTurn } from '../types';

interface ChatFeedProps {
  activeConv: Conversation | null;
  activeConvId: string | null;
  groupedTurns: GroupedTurn[];
  inspectedTurnId: string | null;
  setInspectedTurnId: (id: string | null) => void;
  isStreaming: boolean;
  streamedContent: string;
  inputText: string;
  setInputText: (val: string) => void;
  systemPrompt: string;
  setSystemPrompt: (val: string) => void;
  showSystemPrompt: boolean;
  setShowSystemPrompt: (val: boolean) => void;
  handleSendMessage: (e: FormEvent) => void;
  activeTokensSum: number;
  messageFeedRef: RefObject<HTMLDivElement | null>;
  isLeftCollapsed: boolean;
  setIsLeftCollapsed: (collapsed: boolean) => void;
  isConsoleHidden: boolean;
  setIsConsoleHidden: (hidden: boolean) => void;
  isRightCollapsed: boolean;
  setIsRightCollapsed: (collapsed: boolean) => void;
  provider: 'gemini' | 'mock';
  setProvider: (val: 'gemini' | 'mock') => void;
}

export function ChatFeed({
  activeConv,
  activeConvId,
  groupedTurns,
  inspectedTurnId,
  setInspectedTurnId,
  isStreaming,
  streamedContent,
  inputText,
  setInputText,
  systemPrompt,
  setSystemPrompt,
  showSystemPrompt,
  setShowSystemPrompt,
  handleSendMessage,
  activeTokensSum,
  messageFeedRef,
  isLeftCollapsed,
  setIsLeftCollapsed,
  isConsoleHidden,
  setIsConsoleHidden,
  isRightCollapsed,
  setIsRightCollapsed,
  provider,
  setProvider
}: ChatFeedProps) {
  
  const escapeHtml = (unsafe: string) => {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };

  const renderMessageContent = (text: string) => {
    if (!text) return null;
    const escaped = escapeHtml(text);
    const bolded = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    const lines = bolded.split('\n').join('<br/>');
    return <div dangerouslySetInnerHTML={{ __html: lines }} />;
  };

  return (
    <main className="chat-panel" style={{ flex: 1 }}>
      {/* Header */}
      <header className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <button
            className="devtools-action-btn"
            onClick={() => setIsLeftCollapsed(!isLeftCollapsed)}
            title="Toggle Navigation Sidebar"
            style={{ marginRight: '15px', color: 'var(--nord8)' }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>

          <div className="chat-title-info">
            <h2>{activeConv?.title || (activeConvId ? 'Untitled Conversation' : 'New Conversation')}</h2>
            <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>
              {activeConvId ? `Thread: ${activeConvId}` : 'No Active Thread'}
            </span>
          </div>
        </div>

        <div className="chat-config-controls" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }} htmlFor="provider">Active Model:</label>
          <select
            id="provider"
            className="provider-select"
            value={provider}
            onChange={(e) => setProvider(e.target.value as 'gemini' | 'mock')}
          >
            <option value="gemini">gemini-flash</option>
            <option value="mock">mock (offline)</option>
          </select>

          {/* DevTools Drawer Toggle button */}
          <button
            className="devtools-action-btn"
            onClick={() => setIsConsoleHidden(!isConsoleHidden)}
            title="Toggle DevTools Console Panel"
            style={{ color: 'var(--nord8)', marginLeft: '8px' }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
              <line x1="2" y1="10" x2="22" y2="10"></line>
            </svg>
          </button>

          {/* Right Sidebar toggle button */}
          <button
            className="devtools-action-btn"
            onClick={() => setIsRightCollapsed(!isRightCollapsed)}
            title="Toggle Parameters Sidebar"
            style={{ color: 'var(--nord8)', marginLeft: '8px' }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
          </button>
        </div>
      </header>

      {/* Feed */}
      <section className="message-feed" ref={messageFeedRef}>
        {/* Eviction Summary Banner */}
        {activeConv?.summary && (
          <div style={{
            background: 'rgba(208, 135, 112, 0.1)',
            border: '1px solid var(--nord12)',
            padding: '12px',
            borderRadius: 'var(--radius-md)',
            fontSize: '13px',
            color: 'var(--nord12)',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <span>
                <strong>Memory Eviction Active:</strong> History exceeds 4,000 tokens. Older turns summarized in SQLite.
              </span>
            </div>
            <div style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-color)',
              padding: '10px',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-secondary)'
            }}>
              <strong>Incremental Summary:</strong> {activeConv.summary}
            </div>
          </div>
        )}

        {/* List Grouped Turns */}
        {groupedTurns.map((turn) => {
          const isSelected = turn.id === inspectedTurnId;
          return (
            <div
              key={turn.id}
              onClick={() => setInspectedTurnId(turn.id)}
              style={{
                cursor: 'pointer',
                borderRadius: 'var(--radius-md)',
                padding: '12px 8px',
                transition: 'all var(--transition-fast)',
                border: isSelected ? '1px solid rgba(136, 192, 208, 0.25)' : '1px solid transparent',
                backgroundColor: isSelected ? 'rgba(136, 192, 208, 0.03)' : 'transparent',
                boxShadow: isSelected ? '0 4px 12px rgba(0, 0, 0, 0.15)' : 'none',
                marginBottom: '12px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}
            >
              {/* User Bubble */}
              <article className="message-bubble user">
                <div className="message-header">
                  <span>USER</span>
                  <span>•</span>
                  <span>{new Date(turn.userMessage.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  {isSelected && (
                    <span style={{ color: 'var(--nord8)', background: 'rgba(136, 192, 208, 0.1)', padding: '1px 6px', borderRadius: '10px', fontSize: '10px', fontWeight: 600, marginLeft: '8px' }}>
                      DEBUG ACTIVE
                    </span>
                  )}
                </div>
                <div className="message-content">
                  {turn.userMessage.content}
                </div>
              </article>

              {/* Model response Turn (Sleek Clean Response) */}
              {(turn.modelMessages.length > 0 || turn.finalModelMessage) && (
                <article className="message-bubble model" style={{ marginTop: '8px' }}>
                  <div className="message-header">
                    <span>MODEL</span>
                    <span>•</span>
                    <span>
                      {turn.finalModelMessage
                        ? new Date(turn.finalModelMessage.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                        : 'Processing'}
                    </span>
                    {turn.finalModelMessage?.rag_route === 'RAG' && (
                      <span style={{ color: 'var(--nord14)', background: 'rgba(163, 190, 140, 0.1)', padding: '1px 6px', borderRadius: '10px', fontSize: '10px', fontWeight: 600, marginLeft: '8px' }}>RAG ROUTE</span>
                    )}
                  </div>

                  <div className="message-content">
                    {turn.finalModelMessage && renderMessageContent(turn.finalModelMessage.content || '')}

                    {/* Retrieved citations inside details accordion */}
                    {turn.finalModelMessage?.rag_sources && turn.finalModelMessage.rag_sources.length > 0 && (
                      <details className="metadata-accordion" style={{ marginTop: '15px' }} onClick={(e) => e.stopPropagation()}>
                        <summary className="metadata-header" style={{ listStyle: 'none', display: 'flex', justifyContent: 'space-between', outline: 'none' }}>
                          <span>Retrieved Context Sources ({turn.finalModelMessage.rag_sources.length} chunks matched)</span>
                          <span>▼</span>
                        </summary>
                        <div className="metadata-body">
                          {turn.finalModelMessage.rag_sources.map((src, srcIdx) => (
                            <div key={srcIdx} className="source-item">
                              <div className="source-meta">
                                <strong>{src.document_name} (Chunk {src.chunk_index + 1})</strong>
                                {src.score !== null && (
                                  <span style={{ color: 'var(--nord14)' }}>Score: {src.score.toFixed(2)}</span>
                                )}
                              </div>
                              <div>{src.content}</div>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </article>
              )}
            </div>
          );
        })}

        {/* Active Streaming bubble */}
        {isStreaming && (
          <article className="message-bubble model">
            <div className="message-header">
              <span>MODEL</span>
              <span>•</span>
              <span>Streaming</span>
            </div>
            <div className="message-content">
              {streamedContent ? (
                renderMessageContent(streamedContent)
              ) : (
                <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <span>Coordinating runtime tools / routing query...</span>
                  <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Thinking</span>
                    <span style={{ width: '4px', height: '4px', borderRadius: '50%', backgroundColor: 'var(--text-muted)' }}></span>
                  </div>
                </div>
              )}
            </div>
          </article>
        )}
      </section>

      {/* Input Footer Form */}
      <footer className="chat-footer">
        <form onSubmit={handleSendMessage}>
          {/* System prompt override area */}
          {showSystemPrompt && (
            <div className="system-prompt-panel">
              <label style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)' }}>Modify System Instructions:</label>
              <textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="e.g. You are a specialized assistant that explains clean architecture using analogies..."
              />
            </div>
          )}

          <div className="input-container">
            <textarea
              className="chat-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
              placeholder="Preguntá lo que quieras sobre el sistema de persistencia..."
            />

            <div className="input-controls">
              <button
                type="button"
                className="system-prompt-toggle"
                onClick={() => setShowSystemPrompt(!showSystemPrompt)}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="3"></circle>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
                System Prompt
              </button>

              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span className="token-budget-badge" style={{ color: 'var(--text-secondary)' }}>
                  Active Budget: {activeTokensSum.toLocaleString()} / 4,000 tokens
                </span>
                <button type="submit" className="send-btn" disabled={isStreaming}>Send Run</button>
              </div>
            </div>
          </div>
        </form>
      </footer>
    </main>
  );
}

import type { GroupedTurn } from '../types';

interface ConsoleBottomProps {
  isConsoleHidden: boolean;
  isConsoleCollapsed: boolean;
  setIsConsoleCollapsed: (collapsed: boolean) => void;
  consoleHeight: number;
  toggleConsoleHeight: () => void;
  consoleTab: 'logs' | 'timeline' | 'json' | 'perf';
  setConsoleTab: (tab: 'logs' | 'timeline' | 'json' | 'perf') => void;
  inspectedTurn: GroupedTurn | null;
  logsList: Array<{ timestamp: string; tag: 'info' | 'warn' | 'success' | 'error'; message: string }>;
  evalScores: { faithfulness: number; relevance: number } | null;
  turnLatency: number | null;
  estimatedCost: number;
  totalTurnTokens: number;
  provider: 'gemini' | 'mock';
  temperature: number;
  topK: number;
  topP: number;
  enabledTools: string[];
  systemPrompt: string;
}

export function ConsoleBottom({
  isConsoleHidden,
  isConsoleCollapsed,
  setIsConsoleCollapsed,
  consoleHeight,
  toggleConsoleHeight,
  consoleTab,
  setConsoleTab,
  inspectedTurn,
  logsList,
  evalScores,
  turnLatency,
  estimatedCost,
  totalTurnTokens,
  provider,
  temperature,
  topK,
  topP,
  enabledTools,
  systemPrompt
}: ConsoleBottomProps) {
  if (isConsoleHidden) return null;

  return (
    <section
      className={`devtools-drawer ${isConsoleCollapsed ? 'collapsed' : ''}`}
      style={{ height: isConsoleCollapsed ? '40px' : `${consoleHeight}px` }}
    >
      {/* DevTools Header Tabs */}
      <div className="devtools-header">
        <div className="devtools-tabs">
          <button
            className={`devtools-tab ${consoleTab === 'logs' ? 'active' : ''}`}
            onClick={() => {
              setConsoleTab('logs');
              setIsConsoleCollapsed(false);
            }}
          >
            Terminal Logs
          </button>
          <button
            className={`devtools-tab ${consoleTab === 'timeline' ? 'active' : ''}`}
            onClick={() => {
              setConsoleTab('timeline');
              setIsConsoleCollapsed(false);
            }}
          >
            Execution Steps
          </button>
          <button
            className={`devtools-tab ${consoleTab === 'json' ? 'active' : ''}`}
            onClick={() => {
              setConsoleTab('json');
              setIsConsoleCollapsed(false);
            }}
          >
            Raw Payload JSON
          </button>
          <button
            className={`devtools-tab ${consoleTab === 'perf' ? 'active' : ''}`}
            onClick={() => {
              setConsoleTab('perf');
              setIsConsoleCollapsed(false);
            }}
          >
            Performance & Cost
          </button>
        </div>

        {/* Window Controls */}
        <div className="devtools-header-actions">
          {inspectedTurn ? (
            <span style={{ fontSize: '11px', color: 'var(--nord14)', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--nord14)', display: 'inline-block' }}></span>
              Inspecting Turn Run
            </span>
          ) : (
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 500 }}>
              Select a turn bubble to inspect
            </span>
          )}

          {/* Resize drawer */}
          {!isConsoleCollapsed && (
            <button className="devtools-action-btn" onClick={toggleConsoleHeight} title="Toggle panel height" style={{ color: 'var(--text-secondary)' }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="4 14 10 14 10 20"></polyline>
                <polyline points="20 10 14 10 14 4"></polyline>
              </svg>
            </button>
          )}

          {/* Collapse / Minimize panel drawer */}
          <button className="devtools-action-btn" onClick={() => setIsConsoleCollapsed(!isConsoleCollapsed)} title="Minimize panel drawer" style={{ color: 'var(--text-secondary)' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {isConsoleCollapsed ? (
                <polyline points="4 17 12 9 20 17"></polyline>
              ) : (
                <line x1="5" y1="12" x2="19" y2="12"></line>
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* DevTools Drawer Content */}
      <div className="devtools-content" style={{ display: isConsoleCollapsed ? 'none' : 'block' }}>
        {inspectedTurn ? (
          <>
            {/* TAB 1: Live Logs Console */}
            {consoleTab === 'logs' && (
              <div className="logs-console">
                {logsList.map((log, idx) => (
                  <div className="log-line" key={idx}>
                    <span className="log-timestamp">{log.timestamp}</span>
                    <span className={`log-tag ${log.tag}`}>{log.tag}</span>
                    <span className="log-message" dangerouslySetInnerHTML={{ __html: log.message }}></span>
                  </div>
                ))}
              </div>
            )}

            {/* TAB 2: Execution steps timeline */}
            {consoleTab === 'timeline' && (
              <div className="execution-timeline" style={{ margin: 0, paddingLeft: '15px' }}>
                {/* Step 1: Semantic router decision */}
                <div className="timeline-step completed">
                  <div className="timeline-step-header">
                    <span className="timeline-step-title">1. Routing (Semantic Router)</span>
                    <span className="timeline-step-status" style={{ color: 'var(--text-secondary)' }}>
                      {inspectedTurn.finalModelMessage?.rag_route === 'RAG' ? 'RAG Index Database' : 'Direct LLM Chat'}
                    </span>
                  </div>
                </div>

                {/* Tool executions (if any) */}
                {inspectedTurn.modelMessages.map((m, idx) => {
                  const tc = m.tool_calls?.[0];
                  if (!tc) return null;
                  const toolRes = inspectedTurn.toolMessages.find(t => t.tool_name === tc.name && (t.tool_call_id === m.tool_call_id || idx === 0));

                  return (
                    <div key={m.id} className="timeline-step completed">
                      <div className="timeline-step-header">
                        <span className="timeline-step-title">
                          {idx + 2}. Tool Call Execution: {tc.name}
                        </span>
                        <span className="timeline-step-status" style={{ color: 'var(--text-secondary)' }}>
                          {toolRes ? 'Success' : 'Pending'}
                        </span>
                      </div>
                      <div className="timeline-step-body">
                        <div className="tool-call-block">
                          <div className="tool-args-label" style={{ color: 'var(--nord8)' }}>Arguments:</div>
                          <code>{JSON.stringify(tc.args, null, 2)}</code>
                          {toolRes && (
                            <>
                              <div className="tool-response-label" style={{ color: 'var(--nord14)', marginTop: '8px' }}>Response:</div>
                              <code>{toolRes.content}</code>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}

                {/* Final response generation */}
                {inspectedTurn.finalModelMessage && (
                  <div className="timeline-step completed">
                    <div className="timeline-step-header">
                      <span className="timeline-step-title">
                        {inspectedTurn.modelMessages.length + 2}. Final Text Generation
                      </span>
                      <span className="timeline-step-status" style={{ color: 'var(--text-secondary)' }}>
                        Completed ({inspectedTurn.finalModelMessage.tokens || 0} tokens generated)
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: Raw request/response JSON */}
            {consoleTab === 'json' && (
              <div className="json-viewer-block">
                {JSON.stringify({
                  request: {
                    model: provider === 'gemini' ? 'gemini-flash' : 'mock-provider',
                    query: inspectedTurn.userMessage.content,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    enabledTools: enabledTools,
                    systemPrompt: systemPrompt || 'Default System Prompt'
                  },
                  response: inspectedTurn.finalModelMessage ? {
                    id: inspectedTurn.finalModelMessage.id,
                    content: inspectedTurn.finalModelMessage.content,
                    tokens: inspectedTurn.finalModelMessage.tokens,
                    ragRoute: inspectedTurn.finalModelMessage.rag_route,
                    ragSourcesCount: inspectedTurn.finalModelMessage.rag_sources?.length || 0,
                    created_at: inspectedTurn.finalModelMessage.created_at
                  } : null
                }, null, 2)}
              </div>
            )}

            {/* TAB 4: Performance & Cost */}
            {consoleTab === 'perf' && (
              <div className="perf-metrics-container">
                <div className="metric-card" style={{ background: 'rgba(255,255,255,0.01)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="metric-label-row">
                    <span className="metric-name" style={{ color: 'var(--text-secondary)' }}>Faithfulness (Judge)</span>
                    <span className="metric-value" style={{ color: 'var(--nord14)', fontWeight: 600 }}>
                      {evalScores ? `${evalScores.faithfulness.toFixed(2)}/5.0` : '—'}
                    </span>
                  </div>
                  <div className="metric-progress-bar" style={{ marginTop: '5px' }}>
                    <div className="metric-progress-fill good" style={{ width: evalScores ? `${(evalScores.faithfulness / 5.0) * 100}%` : '0%', background: 'var(--nord14)' }}></div>
                  </div>
                </div>

                <div className="metric-card" style={{ background: 'rgba(255,255,255,0.01)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="metric-label-row">
                    <span className="metric-name" style={{ color: 'var(--text-secondary)' }}>Answer Relevance</span>
                    <span className="metric-value" style={{ color: 'var(--nord14)', fontWeight: 600 }}>
                      {evalScores ? `${evalScores.relevance.toFixed(2)}/5.0` : '—'}
                    </span>
                  </div>
                  <div className="metric-progress-bar" style={{ marginTop: '5px' }}>
                    <div className="metric-progress-fill good" style={{ width: evalScores ? `${(evalScores.relevance / 5.0) * 100}%` : '0%', background: 'var(--nord14)' }}></div>
                  </div>
                </div>

                <div className="metric-card" style={{ background: 'rgba(255,255,255,0.01)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="metric-label-row">
                    <span className="metric-name" style={{ color: 'var(--text-secondary)' }}>Turn Latency</span>
                    <span className="metric-value" style={{ color: 'var(--nord8)', fontWeight: 600 }}>
                      {turnLatency ? `${turnLatency.toFixed(2)} seconds` : '—'}
                    </span>
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '5px', fontWeight: 500 }}>
                    Includes tool calls execution roundtrip times
                  </div>
                </div>

                <div className="metric-card" style={{ background: 'rgba(255,255,255,0.01)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="metric-label-row">
                    <span className="metric-name" style={{ color: 'var(--text-secondary)' }}>Estimated Cost</span>
                    <span className="metric-value" style={{ color: 'var(--nord15)', fontWeight: 600 }}>
                      ${estimatedCost.toFixed(6)} USD
                    </span>
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '5px', fontWeight: 500 }}>
                    Pricing base: {totalTurnTokens} total tokens at Flash pricing
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div style={{ color: 'var(--text-secondary)', fontSize: '13px', textAlign: 'center', marginTop: '30px' }}>
            No active turn run selected. Click on a message in the chat feed to inspect its parameters.
          </div>
        )}
      </div>
    </section>
  );
}

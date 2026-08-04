import type { RefObject, ChangeEvent, FormEvent } from 'react';
import type { RAGDoc } from '../types';

interface SidebarRightProps {
  isRightCollapsed: boolean;
  temperature: number;
  setTemperature: (val: number) => void;
  topK: number;
  setTopK: (val: number) => void;
  topP: number;
  setTopP: (val: number) => void;
  enabledTools: string[];
  handleToolCheckboxChange: (toolName: string) => void;
  fileInputRef: RefObject<HTMLInputElement | null>;
  handleFileUploadClick: () => void;
  handleFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  isUploading: boolean;
  uploadError: string | null;
  docs: RAGDoc[];
  handleDeleteDocument: (name: string) => void;
  sandboxQuery: string;
  setSandboxQuery: (val: string) => void;
  handleSandboxSearch: (e: FormEvent) => void;
  sandboxLoading: boolean;
  sandboxResults: any[];
}

export function SidebarRight({
  isRightCollapsed,
  temperature,
  setTemperature,
  topK,
  setTopK,
  topP,
  setTopP,
  enabledTools,
  handleToolCheckboxChange,
  fileInputRef,
  handleFileUploadClick,
  handleFileChange,
  isUploading,
  uploadError,
  docs,
  handleDeleteDocument,
  sandboxQuery,
  setSandboxQuery,
  handleSandboxSearch,
  sandboxLoading,
  sandboxResults
}: SidebarRightProps) {
  return (
    <aside className={`sidebar-right ${isRightCollapsed ? 'collapsed' : ''}`} id="sidebar-right">
      <div className="sidebar-right-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 20h9"></path>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
          </svg>
          Runtime Config
        </h3>
      </div>

      <div className="sidebar-right-content" style={{ gap: '15px', padding: '15px' }}>
        {/* Sliders parameters block */}
        <div className="config-section">
          <div className="config-section-title">Model Parameters</div>

          {/* Temperature */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="slider-title">Temperature</span>
              <span className="slider-val">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="slider-input"
              min="0"
              max="2"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>

          {/* Top-K */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="slider-title">Top-K</span>
              <span className="slider-val">{topK}</span>
            </div>
            <input
              type="range"
              className="slider-input"
              min="1"
              max="100"
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
            />
          </div>

          {/* Top-P */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="slider-title">Top-P</span>
              <span className="slider-val">{topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="slider-input"
              min="0"
              max="1"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
            />
          </div>
        </div>

        {/* Toggle Tools block */}
        <div className="config-section">
          <div className="config-section-title">Enabled Runtime Tools</div>
          <div className="tools-checkbox-grid">
            <label className="tool-checkbox-label">
              <input
                type="checkbox"
                checked={enabledTools.includes('query_database')}
                onChange={() => handleToolCheckboxChange('query_database')}
              />
              <code>query_database</code> (SQLite DB)
            </label>

            <label className="tool-checkbox-label">
              <input
                type="checkbox"
                checked={enabledTools.includes('run_shell_command')}
                onChange={() => handleToolCheckboxChange('run_shell_command')}
              />
              <code>run_shell_command</code> (Shell execution)
            </label>

            <label className="tool-checkbox-label">
              <input
                type="checkbox"
                checked={enabledTools.includes('read_file')}
                onChange={() => handleToolCheckboxChange('read_file')}
              />
              <code>read_file</code> (FS Reader)
            </label>
          </div>
        </div>

        {/* RAG Ingestion block */}
        <div className="config-section">
          <div className="config-section-title">RAG Ingestion Database</div>

          <div className="rag-upload-zone" onClick={handleFileUploadClick}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--nord8)" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>
              {isUploading ? 'Uploading file...' : 'Ingest local file'}
            </span>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
              accept=".txt,.md,.markdown"
            />
          </div>

          {uploadError && (
            <div style={{ color: 'var(--color-error)', fontSize: '11px', textAlign: 'center' }}>
              {uploadError}
            </div>
          )}

          <div className="rag-doc-list" style={{ marginTop: '5px' }}>
            {docs.map((doc) => (
              <div
                className="rag-doc-item"
                key={doc.id}
                style={{ padding: '8px', borderColor: 'var(--border-color)', backgroundColor: 'rgba(0,0,0,0.1)' }}
              >
                <div className="doc-icon" style={{ color: 'var(--nord8)' }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                  </svg>
                </div>
                <div className="doc-info" style={{ flex: 1 }}>
                  <div className="doc-name" style={{ fontSize: '12px', color: 'var(--text-primary)' }} title={doc.name}>
                    {doc.name}
                  </div>
                  <div className="doc-meta" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
                <button
                  className="doc-action-btn"
                  title="Delete document"
                  onClick={() => handleDeleteDocument(doc.name)}
                  style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--nord11)' }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  </svg>
                </button>
              </div>
            ))}

            {docs.length === 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '10px' }}>
                No documents ingested.
              </div>
            )}
          </div>
        </div>

        {/* Sandbox block */}
        <div className="rag-sandbox" style={{ marginTop: '5px' }}>
          <div className="config-section-title">RAG Vector Sandbox</div>
          <form onSubmit={handleSandboxSearch} className="rag-sandbox-input-container" style={{ display: 'flex', gap: '4px' }}>
            <input
              type="text"
              className="rag-sandbox-input"
              placeholder="Query index directly..."
              value={sandboxQuery}
              onChange={(e) => setSandboxQuery(e.target.value)}
              style={{ flex: 1 }}
            />
            <button type="submit" className="rag-sandbox-btn" disabled={sandboxLoading}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
            </button>
          </form>

          {sandboxResults.length > 0 && (
            <div
              style={{
                maxHeight: '120px',
                overflowY: 'auto',
                fontSize: '11px',
                background: 'var(--bg-primary)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px',
                border: '1px solid var(--border-color)',
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                marginTop: '10px'
              }}
            >
              {sandboxResults.map((chunk, idx) => (
                <div key={chunk.id || idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontWeight: 600 }}>
                    <span>{chunk.document_name}</span>
                    {chunk.score !== null && (
                      <span style={{ color: 'var(--nord14)' }}>{chunk.score.toFixed(2)}</span>
                    )}
                  </div>
                  <div style={{ color: 'var(--text-secondary)', marginTop: '2px' }}>{chunk.content}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}

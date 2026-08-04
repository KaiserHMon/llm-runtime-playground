import type { Conversation } from '../types';

interface SidebarLeftProps {
  conversations: Conversation[];
  activeConvId: string | null;
  setActiveConvId: (id: string | null) => void;
  handleCreateConversation: () => void;
  isLeftCollapsed: boolean;
}

export function SidebarLeft({
  conversations,
  activeConvId,
  setActiveConvId,
  handleCreateConversation,
  isLeftCollapsed
}: SidebarLeftProps) {
  return (
    <aside className={`sidebar ${isLeftCollapsed ? 'collapsed' : ''}`} id="sidebar-left">
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={handleCreateConversation}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          New Chat
        </button>
      </div>

      <div className="conversation-list">
        {conversations.map((conv) => (
          <div
            key={conv.id}
            className={`conversation-item ${activeConvId === conv.id ? 'active' : ''}`}
            onClick={() => setActiveConvId(conv.id)}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="conversation-title">{conv.title || 'Untitled Conversation'}</div>
              <div className="conversation-meta" style={{ color: 'var(--text-muted)' }}>
                ID: {conv.id.substring(0, 8)}... • {new Date(conv.updated_at).toLocaleDateString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

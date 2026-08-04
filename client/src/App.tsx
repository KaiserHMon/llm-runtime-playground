import { useState, useEffect, useRef } from 'react';
import { useChat } from './hooks/useChat';
import { SidebarLeft } from './components/SidebarLeft';
import { SidebarRight } from './components/SidebarRight';
import { ChatFeed } from './components/ChatFeed';
import { ConsoleBottom } from './components/ConsoleBottom';

export default function App() {
  const chat = useChat();

  // Layout states (left/right sidebars and bottom console heights)
  const [isLeftCollapsed, setIsLeftCollapsed] = useState(false);
  const [isRightCollapsed, setIsRightCollapsed] = useState(false);
  const [isConsoleHidden, setIsConsoleHidden] = useState(false);
  const [isConsoleCollapsed, setIsConsoleCollapsed] = useState(false);
  const [consoleHeight, setConsoleHeight] = useState(280);
  const [consoleTab, setConsoleTab] = useState<'logs' | 'timeline' | 'json' | 'perf'>('logs');

  const messageFeedRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on updates
  const scrollToBottom = () => {
    if (messageFeedRef.current) {
      messageFeedRef.current.scrollTop = messageFeedRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [chat.messages, chat.streamedContent, chat.isStreaming]);

  const toggleConsoleHeight = () => {
    setConsoleHeight((prev) => (prev === 280 ? 500 : 280));
  };

  return (
    <div className="app-container">
      {/* 1. LEFT SIDEBAR: Conversations History */}
      <SidebarLeft
        conversations={chat.conversations}
        activeConvId={chat.activeConvId}
        setActiveConvId={chat.setActiveConvId}
        handleCreateConversation={chat.handleCreateConversation}
        isLeftCollapsed={isLeftCollapsed}
      />

      {/* CENTER WORKSPACE: Chat Feed + Bottom Inspector Console Drawer */}
      <div className="workspace-container">
        <ChatFeed
          activeConv={chat.activeConv}
          activeConvId={chat.activeConvId}
          groupedTurns={chat.groupedTurns}
          inspectedTurnId={chat.inspectedTurnId}
          setInspectedTurnId={chat.setInspectedTurnId}
          isStreaming={chat.isStreaming}
          streamedContent={chat.streamedContent}
          inputText={chat.inputText}
          setInputText={chat.setInputText}
          systemPrompt={chat.systemPrompt}
          setSystemPrompt={chat.setSystemPrompt}
          showSystemPrompt={chat.showSystemPrompt}
          setShowSystemPrompt={chat.setShowSystemPrompt}
          handleSendMessage={chat.handleSendMessage}
          activeTokensSum={chat.activeTokensSum}
          messageFeedRef={messageFeedRef}
          isLeftCollapsed={isLeftCollapsed}
          setIsLeftCollapsed={setIsLeftCollapsed}
          isConsoleHidden={isConsoleHidden}
          setIsConsoleHidden={setIsConsoleHidden}
          isRightCollapsed={isRightCollapsed}
          setIsRightCollapsed={setIsRightCollapsed}
          provider={chat.provider}
          setProvider={chat.setProvider}
        />

        {/* 3. DEVTOOLS BOTTOM DRAWER (Logs, steps, JSON specs) */}
        <ConsoleBottom
          isConsoleHidden={isConsoleHidden}
          isConsoleCollapsed={isConsoleCollapsed}
          setIsConsoleCollapsed={setIsConsoleCollapsed}
          consoleHeight={consoleHeight}
          toggleConsoleHeight={toggleConsoleHeight}
          consoleTab={consoleTab}
          setConsoleTab={setConsoleTab}
          inspectedTurn={chat.inspectedTurn}
          logsList={chat.logsList}
          evalScores={chat.evalScores}
          turnLatency={chat.turnLatency}
          estimatedCost={chat.estimatedCost}
          totalTurnTokens={chat.totalTurnTokens}
          provider={chat.provider}
          temperature={chat.temperature}
          topK={chat.topK}
          topP={chat.topP}
          enabledTools={chat.enabledTools}
          systemPrompt={chat.systemPrompt}
        />
      </div>

      {/* 4. RIGHT SIDEBAR: Parameters & RAG configuration */}
      <SidebarRight
        isRightCollapsed={isRightCollapsed}
        temperature={chat.temperature}
        setTemperature={chat.setTemperature}
        topK={chat.topK}
        setTopK={chat.setTopK}
        topP={chat.topP}
        setTopP={chat.setTopP}
        enabledTools={chat.enabledTools}
        handleToolCheckboxChange={chat.handleToolCheckboxChange}
        fileInputRef={chat.fileInputRef}
        handleFileUploadClick={chat.handleFileUploadClick}
        handleFileChange={chat.handleFileChange}
        isUploading={chat.isUploading}
        uploadError={chat.uploadError}
        docs={chat.docs}
        handleDeleteDocument={chat.handleDeleteDocument}
      />
    </div>
  );
}

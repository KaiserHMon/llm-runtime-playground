export interface Message {
  id: string;
  role: 'user' | 'model' | 'system' | 'tool';
  content: string | null;
  tokens: number | null;
  tool_calls: Array<{ name: string; args: any }> | null;
  tool_name: string | null;
  tool_call_id?: string | null;
  parts: any[] | null;
  rag_route: string | null;
  rag_sources: Array<{ document_name: string; chunk_index: number; content: string; score: number | null }> | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string | null;
  summary: string | null;
  last_summarized_message_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface RAGDoc {
  id: string;
  name: string;
  created_at: string;
}

export interface GroupedTurn {
  id: string; // user message id
  userMessage: Message;
  modelMessages: Message[]; // intermediate model turns with tool calls
  toolMessages: Message[];  // intermediate tool response turns
  finalModelMessage: Message | null; // final text response message
}

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types
from app.models.chat import Message as DBMessage, MessageRole
from app.services.llm.gemini import GeminiProvider
from app.services.embedding.gemini import GeminiEmbeddingProvider
from app.services.llm.base import ToolCall
import app.services.tools

@pytest.mark.asyncio
async def test_gemini_provider_init_and_client_error(monkeypatch):
    # Test client initialization when API key is missing
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "")
    
    provider = GeminiProvider()
    assert provider.model_id == "gemini-flash-lite-latest"
    
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        _ = provider.client

@pytest.mark.asyncio
async def test_gemini_provider_client_success(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    with patch("google.genai.Client") as mock_client_cls:
        client = provider.client
        assert client is not None
        mock_client_cls.assert_called_once_with(api_key="test_key")

def test_gemini_provider_get_tools():
    provider = GeminiProvider()
    
    # Check default tools retrieval (should include get_weather, get_current_datetime, list_conversations)
    tools = provider._get_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], types.Tool)
    assert len(tools[0].function_declarations) > 0
    
    # Check tool filtering
    filtered_tools = provider._get_tools(enabled_tools=["get_weather"])
    assert len(filtered_tools) == 1
    funcs = filtered_tools[0].function_declarations
    assert len(funcs) == 1
    assert funcs[0].name == "get_weather"

    # Register temporary function to test parameter types (int, float, bool, str)
    def dummy_tool(a: int, b: float, c: bool, d: str):
        pass
    
    app.services.tools.registry.register(dummy_tool)
    try:
        tools = provider._get_tools(enabled_tools=["dummy_tool"])
        assert len(tools) == 1
        params = tools[0].function_declarations[0].parameters.properties
        assert params["a"].type == types.Type.INTEGER
        assert params["b"].type == types.Type.NUMBER
        assert params["c"].type == types.Type.BOOLEAN
        assert params["d"].type == types.Type.STRING
    finally:
        # cleanup dummy_tool
        app.services.tools.registry._tools.pop("dummy_tool", None)

def test_gemini_provider_build_context():
    provider = GeminiProvider()
    
    # Prepare mock DB messages
    messages = [
        DBMessage(
            conversation_id="c1",
            role=MessageRole.USER,
            content="Hola"
        ),
        DBMessage(
            conversation_id="c1",
            role=MessageRole.MODEL,
            content="Hola, ¿en qué te ayudo?",
            tool_calls=[{"name": "get_weather", "args": {"location": "Buenos Aires"}}]
        ),
        DBMessage(
            conversation_id="c1",
            role=MessageRole.TOOL,
            content="72F and sunny",
            tool_name="get_weather"
        ),
        # A message with pre-existing parts list (representing multi-part or structured content)
        DBMessage(
            conversation_id="c1",
            role=MessageRole.USER,
            parts=[{"text": "Pregunta final"}]
        ),
        # A message with system role to test filtering / continue statement (line 78)
        DBMessage(
            conversation_id="c1",
            role=MessageRole.SYSTEM,
            content="System prompt"
        )
    ]
    
    contents = provider._build_context(messages)
    
    # Should be 4, as SYSTEM message is skipped (line 78)
    assert len(contents) == 4
    
    # Assert roles mappings
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hola"
    
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Hola, ¿en qué te ayudo?"
    assert contents[1].parts[1].function_call.name == "get_weather"
    assert contents[1].parts[1].function_call.args == {"location": "Buenos Aires"}
    
    # MessageRole.TOOL maps to user. Since content is present, it creates a text part first,
    # then a function response part.
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "72F and sunny"
    assert contents[2].parts[1].function_response.name == "get_weather"
    assert contents[2].parts[1].function_response.response == {"result": "72F and sunny"}
    
    assert contents[3].role == "user"
    assert contents[3].parts[0].text == "Pregunta final"

@pytest.mark.asyncio
async def test_gemini_provider_generate_response(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = "Respuesta del modelo"
    
    # Mock function call
    mock_fc = MagicMock()
    mock_fc.name = "get_weather"
    mock_fc.args = {"location": "Mar del Plata"}
    mock_response.function_calls = [mock_fc]
    
    # Mock candidates/parts
    mock_part = MagicMock()
    mock_part.model_dump.return_value = {"text": "Respuesta del modelo"}
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    
    # Mock usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 15
    mock_response.usage_metadata = mock_usage
    
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    with patch("google.genai.Client", return_value=mock_client):
        history = [DBMessage(conversation_id="c1", role=MessageRole.USER, content="¿Qué tiempo hace?")]
        response = await provider.generate_response(
            history=history,
            summary="Resumen previo",
            rag_context="Documento de soporte",
            temperature=0.5,
            top_k=20,
            top_p=0.9,
            enabled_tools=["get_weather"]
        )
        
        assert response.content == "Respuesta del modelo"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0] == ToolCall(name="get_weather", args={"location": "Mar del Plata"})
        assert response.parts == [{"text": "Respuesta del modelo"}]
        assert response.input_tokens == 10
        assert response.output_tokens == 15

@pytest.mark.asyncio
async def test_gemini_provider_generate_response_stream(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    # Create chunks to yield in async generator
    chunk1 = MagicMock()
    chunk1.text = "Parte 1"
    chunk2 = MagicMock()
    chunk2.text = "Parte 2"
    
    async def async_generator():
        yield chunk1
        yield chunk2
        
    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=async_generator())
    
    with patch("google.genai.Client", return_value=mock_client):
        history = [DBMessage(conversation_id="c1", role=MessageRole.USER, content="Streaming")]
        stream = provider.generate_response_stream(
            history=history,
            summary="Previo",
            rag_context="RAG context data"
        )
        
        results = []
        async for text in stream:
            results.append(text)
            
        assert results == ["Parte 1", "Parte 2"]

@pytest.mark.asyncio
async def test_gemini_provider_count_tokens(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    mock_response = MagicMock()
    mock_response.total_tokens = 42
    
    mock_client = MagicMock()
    mock_client.aio.models.count_tokens = AsyncMock(return_value=mock_response)
    
    with patch("google.genai.Client", return_value=mock_client):
        # Empty text
        assert await provider.count_tokens("") == 0
        
        # Valid text
        tokens = await provider.count_tokens("Hola mundo")
        assert tokens == 42
        mock_client.aio.models.count_tokens.assert_called_with(
            model="gemini-flash-lite-latest",
            contents="Hola mundo"
        )

@pytest.mark.asyncio
async def test_gemini_provider_summarize_messages(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    mock_response = MagicMock()
    mock_response.text = "Resumen consolidado"
    
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    with patch("google.genai.Client", return_value=mock_client):
        # Empty messages
        assert await provider.summarize_messages("resumen", []) == "resumen"
        
        # With messages and previous summary
        messages = [
            DBMessage(conversation_id="c1", role=MessageRole.USER, content="Pregunta"),
            DBMessage(conversation_id="c1", role=MessageRole.MODEL, content="Respuesta", tool_calls=[{"name": "foo"}], tool_name="foo")
        ]
        
        summary = await provider.summarize_messages("resumen", messages)
        assert summary == "Resumen consolidado"
        
        # Without previous summary
        summary_no_prev = await provider.summarize_messages(None, messages)
        assert summary_no_prev == "Resumen consolidado"

def test_gemini_provider_format_tool_response():
    provider = GeminiProvider()
    formatted = provider.format_tool_response("get_weather", "Soleado")
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert "function_response" in formatted[0]
    assert formatted[0]["function_response"]["name"] == "get_weather"

@pytest.mark.asyncio
async def test_gemini_provider_route_message_success(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    mock_response = MagicMock()
    mock_response.text = '{"route": "RAG"}'
    
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    with patch("google.genai.Client", return_value=mock_client):
        # Testing with non-empty history to cover lines 272-279 and 289
        history = [
            DBMessage(conversation_id="c1", role=MessageRole.USER, content="¿Dónde está mi tarjeta de crédito?"),
            DBMessage(conversation_id="c1", role=MessageRole.MODEL, content="No la tengo"),
            DBMessage(conversation_id="c1", role=MessageRole.TOOL, content="resultado", tool_name="check_card")
        ]
        route = await provider.route_message("¿Dónde está el código de acceso?", history=history)
        assert route == "RAG"

@pytest.mark.asyncio
async def test_gemini_provider_route_message_empty_response(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    mock_response = MagicMock()
    mock_response.text = ""
    
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    with patch("google.genai.Client", return_value=mock_client):
        route = await provider.route_message("Hola")
        assert route == "CHAT"

@pytest.mark.asyncio
async def test_gemini_provider_route_message_exception(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    provider = GeminiProvider()
    
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("API failure"))
    
    with patch("google.genai.Client", return_value=mock_client):
        # Exception should fall back to CHAT
        route = await provider.route_message("¿Dónde está el código de acceso?")
        assert route == "CHAT"

@pytest.mark.asyncio
async def test_gemini_embedding_provider_error(monkeypatch):
    # Test initialization issues when GEMINI_API_KEY is missing
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "")
    emb_provider = GeminiEmbeddingProvider()
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        _ = emb_provider.client

@pytest.mark.asyncio
async def test_gemini_embedding_provider_success(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    emb_provider = GeminiEmbeddingProvider()
    
    # 1. Success case with gemini-embedding-2
    mock_emb_val = MagicMock()
    mock_emb_val.values = [0.1] * 768
    mock_response = MagicMock()
    mock_response.embeddings = [mock_emb_val]
    
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)
    
    with patch("google.genai.Client", return_value=mock_client):
        # Empty text returns empty list
        assert await emb_provider.get_embedding("") == []
        
        # Standard text
        emb = await emb_provider.get_embedding("Hola")
        assert emb == [0.1] * 768
        mock_client.aio.models.embed_content.assert_called_with(
            model="gemini-embedding-2",
            contents="Hola"
        )

@pytest.mark.asyncio
async def test_gemini_embedding_provider_fallback(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    emb_provider = GeminiEmbeddingProvider()
    
    mock_emb_val = MagicMock()
    mock_emb_val.values = [0.1] * 768
    mock_response = MagicMock()
    mock_response.embeddings = [mock_emb_val]
    
    # 2. Fallback case where gemini-embedding-2 raises exception, falls back to gemini-embedding-001
    mock_client = MagicMock()
    # First call raises Exception, second call returns mock_response
    mock_client.aio.models.embed_content = AsyncMock(side_effect=[Exception("model not found"), mock_response])
    
    with patch("google.genai.Client", return_value=mock_client):
        emb = await emb_provider.get_embedding("Hola")
        assert emb == [0.1] * 768
        assert mock_client.aio.models.embed_content.call_count == 2

@pytest.mark.asyncio
async def test_gemini_embedding_provider_failure(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test_key")
    emb_provider = GeminiEmbeddingProvider()
    
    # 3. Complete failure case
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(side_effect=[Exception("model not found"), Exception("another model not found")])
    
    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(ValueError, match="Failed to retrieve embedding values"):
            await emb_provider.get_embedding("Hola")

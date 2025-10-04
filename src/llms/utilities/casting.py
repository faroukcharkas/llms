from typing import Any
from llms.types.messages import ModelMessage, SystemModelMessage, UserModelMessage, AssistantModelMessage
from llms.types.parts import (
    TextPart, ImagePart, FilePart, ReasoningPart, ToolCallPart, ToolResultPart,
    ContentPart, PartType
)
from llms.types.enums import Role


def cast_part_to_openai_content(part: ContentPart) -> dict[str, Any]:
    """Convert an internal Part to OpenAI content format."""
    match part.type:
        case PartType.TEXT:
            return {
                "type": "text",
                "text": part.text
            }
        case PartType.IMAGE:
            return {
                "type": "image_url",
                "image_url": {
                    "url": part.image,
                    **({"detail": part.provider_options.get("detail")} if "detail" in part.provider_options else {})
                }
            }
        case PartType.FILE:
            # OpenAI doesn't have native file support in this way, treating as data URL
            data_url = f"data:{part.media_type};base64,{part.data}"
            return {
                "type": "image_url" if part.media_type.startswith("image/") else "text",
                "image_url": {"url": data_url} if part.media_type.startswith("image/") else {},
                "text": f"File: {part.filename}" if not part.media_type.startswith("image/") else ""
            }
        case PartType.REASONING:
            # Reasoning might be represented as text in the request
            return {
                "type": "text",
                "text": f"[Reasoning] {part.text}"
            }
        case PartType.TOOL_CALL:
            # Tool calls are handled separately in OpenAI API
            return {
                "type": "tool_call",
                "id": part.tool_call_id,
                "function": {
                    "name": part.tool_name,
                    "arguments": part.input
                }
            }
        case PartType.TOOL_RESULT:
            # Tool results are handled separately
            return {
                "type": "tool_result",
                "tool_call_id": part.tool_call_id,
                "content": part.output
            }
        case _:
            raise ValueError(f"Unknown part type: {part.type}")


def cast_message_to_openai(message: ModelMessage) -> dict[str, Any]:
    """Convert an internal ModelMessage to OpenAI message format."""
    openai_message: dict[str, Any] = {
        "role": message.role.value
    }
    
    # Handle content based on type
    if isinstance(message.content, str):
        openai_message["content"] = message.content
    elif isinstance(message.content, list):
        # Convert list of parts to OpenAI content format
        content_items = []
        tool_calls = []
        
        for part in message.content:
            if part.type == PartType.TOOL_CALL:
                tool_calls.append({
                    "id": part.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": part.tool_name,
                        "arguments": str(part.input)
                    }
                })
            else:
                openai_content = cast_part_to_openai_content(part)
                if openai_content.get("type") in ["text", "image_url"]:
                    content_items.append(openai_content)
        
        if content_items:
            openai_message["content"] = content_items if len(content_items) > 1 else content_items[0].get("text", content_items[0])
        
        if tool_calls:
            openai_message["tool_calls"] = tool_calls
    
    return openai_message


def cast_openai_response_to_parts(response: Any) -> list[ContentPart]:
    """Convert OpenAI response to internal Parts format."""
    parts: list[ContentPart] = []
    
    # Handle the response based on its structure
    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]
        message = choice.message
        
        # Handle text content
        if hasattr(message, 'content') and message.content:
            parts.append(TextPart(
                type=PartType.TEXT,
                text=message.content,
                provider_options={}
            ))
        
        # Handle tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append(ToolCallPart(
                    type=PartType.TOOL_CALL,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    input=tool_call.function.arguments,
                    provider_options={},
                    provider_executed=None
                ))
        
        # Handle reasoning (if present in extended thinking response)
        if hasattr(message, 'reasoning') and message.reasoning:
            parts.append(ReasoningPart(
                type=PartType.REASONING,
                text=message.reasoning,
                provider_options={}
            ))
    
    # If no parts were extracted, add an empty text part
    if not parts:
        parts.append(TextPart(
            type=PartType.TEXT,
            text="",
            provider_options={}
        ))
    
    return parts


def cast_parts_to_text(parts: list[ContentPart]) -> str:
    """Convert a list of Parts to a single text string."""
    text_parts = []
    
    for part in parts:
        match part.type:
            case PartType.TEXT:
                text_parts.append(part.text)
            case PartType.REASONING:
                text_parts.append(f"[Reasoning: {part.text}]")
            case PartType.TOOL_CALL:
                text_parts.append(f"[Tool Call: {part.tool_name}]")
            case PartType.TOOL_RESULT:
                text_parts.append(f"[Tool Result: {part.output}]")
            case PartType.IMAGE:
                text_parts.append("[Image]")
            case PartType.FILE:
                text_parts.append(f"[File: {part.filename or 'unnamed'}]")
    
    return " ".join(text_parts).strip()


def cast_part_to_anthropic_content(part: ContentPart) -> dict[str, Any]:
    """Convert an internal Part to Anthropic content format."""
    match part.type:
        case PartType.TEXT:
            return {
                "type": "text",
                "text": part.text
            }
        case PartType.IMAGE:
            # Anthropic expects image data or source
            if part.image.startswith("http://") or part.image.startswith("https://"):
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": part.image
                    }
                }
            else:
                # Assume base64 encoded data
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type or "image/jpeg",
                        "data": part.image
                    }
                }
        case PartType.FILE:
            # Anthropic supports document/file content
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": part.data
                }
            }
        case PartType.TOOL_CALL:
            return {
                "type": "tool_use",
                "id": part.tool_call_id,
                "name": part.tool_name,
                "input": part.input
            }
        case PartType.TOOL_RESULT:
            return {
                "type": "tool_result",
                "tool_use_id": part.tool_call_id,
                "content": part.output
            }
        case PartType.REASONING:
            # Anthropic doesn't have native reasoning type, treat as text
            return {
                "type": "text",
                "text": part.text
            }
        case _:
            raise ValueError(f"Unknown part type: {part.type}")


def cast_message_to_anthropic(message: ModelMessage) -> dict[str, Any]:
    """Convert an internal ModelMessage to Anthropic message format."""
    anthropic_message: dict[str, Any] = {
        "role": message.role.value
    }
    
    # Handle content based on type
    if isinstance(message.content, str):
        anthropic_message["content"] = message.content
    elif isinstance(message.content, list):
        # Convert list of parts to Anthropic content format
        content_items = []
        
        for part in message.content:
            anthropic_content = cast_part_to_anthropic_content(part)
            content_items.append(anthropic_content)
        
        anthropic_message["content"] = content_items
    
    return anthropic_message


def cast_anthropic_response_to_parts(response: Any) -> list[ContentPart]:
    """Convert Anthropic response to internal Parts format."""
    parts: list[ContentPart] = []
    
    # Handle the response content
    if hasattr(response, 'content') and response.content:
        for content_block in response.content:
            # Handle text content
            if hasattr(content_block, 'type') and content_block.type == 'text':
                parts.append(TextPart(
                    type=PartType.TEXT,
                    text=content_block.text,
                    provider_options={}
                ))
            
            # Handle tool use
            elif hasattr(content_block, 'type') and content_block.type == 'tool_use':
                parts.append(ToolCallPart(
                    type=PartType.TOOL_CALL,
                    tool_call_id=content_block.id,
                    tool_name=content_block.name,
                    input=content_block.input,
                    provider_options={},
                    provider_executed=None
                ))
    
    # If no parts were extracted, add an empty text part
    if not parts:
        parts.append(TextPart(
            type=PartType.TEXT,
            text="",
            provider_options={}
        ))
    
    return parts


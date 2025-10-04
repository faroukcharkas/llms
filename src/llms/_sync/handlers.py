from openai import OpenAI
from anthropic import Anthropic
from llms.types.messages import ModelMessage
from llms.types.results import GenerateTextResult
from llms.utilities.casting import (
    cast_message_to_openai,
    cast_openai_response_to_parts,
    cast_message_to_anthropic,
    cast_anthropic_response_to_parts,
    cast_parts_to_text
)


def handle_openai_generate_text(
    openai_client: OpenAI, 
    model_name: str, 
    messages: list[ModelMessage]
) -> GenerateTextResult:
    """
    Handle OpenAI text generation by converting internal messages to OpenAI format,
    calling the API, and converting the response back to internal Parts.
    
    Args:
        openai_client: The OpenAI client instance
        model_name: The name of the OpenAI model to use
        messages: List of internal ModelMessage objects
        
    Returns:
        GenerateTextResult containing both the text response and structured parts
    """
    # Convert internal ModelMessage format to OpenAI format
    openai_messages = [cast_message_to_openai(msg) for msg in messages]
    
    # Call OpenAI chat completions API
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=openai_messages
    )
    
    # Convert OpenAI response to internal Parts format
    parts = cast_openai_response_to_parts(response)
    
    # Convert parts to text for backward compatibility
    text = cast_parts_to_text(parts)
    
    return GenerateTextResult(text=text, parts=parts)

def handle_anthropic_generate_text(
    anthropic_client: Anthropic,
    model_name: str,
    messages: list[ModelMessage]
) -> GenerateTextResult:
    """
    Handle Anthropic text generation by converting internal messages to Anthropic format,
    calling the API, and converting the response back to internal Parts.
    
    Args:
        anthropic_client: The Anthropic client instance
        model_name: The name of the Anthropic model to use
        messages: List of internal ModelMessage objects
        
    Returns:
        GenerateTextResult containing both the text response and structured parts
    """
    # Convert internal ModelMessage format to Anthropic format
    # Note: Anthropic separates system messages from the messages array
    system_messages = [msg for msg in messages if msg.role.value == "system"]
    non_system_messages = [msg for msg in messages if msg.role.value != "system"]
    
    anthropic_messages = [cast_message_to_anthropic(msg) for msg in non_system_messages]
    
    # Build API call parameters
    api_params = {
        "model": model_name,
        "max_tokens": 1024,  # Required parameter for Anthropic
        "messages": anthropic_messages
    }
    
    # Add system message if present
    if system_messages:
        api_params["system"] = system_messages[0].content
    
    # Call Anthropic messages API
    response = anthropic_client.messages.create(**api_params)
    
    # Convert Anthropic response to internal Parts format
    parts = cast_anthropic_response_to_parts(response)
    
    # Convert parts to text for backward compatibility
    text = cast_parts_to_text(parts)
    
    return GenerateTextResult(text=text, parts=parts)

def handle_fireworks_generate_text(
    fireworks_client: OpenAI,
    model_name: str,
    messages: list[ModelMessage]
) -> GenerateTextResult:
    """
    Handle Fireworks text generation by converting internal messages to OpenAI format,
    calling the Fireworks API (which is OpenAI-compatible), and converting the response 
    back to internal Parts.
    
    Args:
        fireworks_client: The OpenAI client instance configured for Fireworks API
        model_name: The name of the Fireworks model to use
        messages: List of internal ModelMessage objects
        
    Returns:
        GenerateTextResult containing both the text response and structured parts
    """
    # Convert internal ModelMessage format to OpenAI format
    # Fireworks uses OpenAI-compatible API format
    openai_messages = [cast_message_to_openai(msg) for msg in messages]
    
    # Call Fireworks chat completions API (OpenAI-compatible)
    response = fireworks_client.chat.completions.create(
        model="accounts/fireworks/models/" + model_name,
        messages=openai_messages
    )
    
    # Convert response to internal Parts format
    parts = cast_openai_response_to_parts(response)
    
    # Convert parts to text for backward compatibility
    text = cast_parts_to_text(parts)
    
    return GenerateTextResult(text=text, parts=parts)

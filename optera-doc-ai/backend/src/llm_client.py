import os
import json
import time
import requests
import base64
import logging
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def call_openrouter(model: str, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Calls OpenRouter API and returns the text and usage stats."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    start_time = time.time()
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    time_taken = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    
    text_content = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    
    # Estimate cost based on model (placeholder values, actuals depend on openrouter pricing)
    # Using 0 as placeholder for open-source model cost if not provided by API
    cost = 0.0001 
    
    stats = {
        "model_used": model,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "time_taken": round(time_taken, 2),
        "estimated_cost": cost
    }
    
    return text_content, stats

def call_google_gemini(model: str, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Calls the native Google Gemini API and returns the text and usage stats."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    time_taken = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    
    text_content = result['candidates'][0]['content']['parts'][0]['text']
    usage = result.get('usageMetadata', {})
    
    stats = {
        "model_used": model,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "time_taken": round(time_taken, 2),
        "estimated_cost": 0.0001
    }
    
    return text_content, stats

def call_google_gemini_vision(image_path: str, system_prompt: str, model_name: str = "gemini-3-flash-preview") -> Tuple[str, Dict[str, Any]]:
    """Calls the native Google Gemini API directly with an image."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    url = f"https://generativelanguage.googleapis.com/v1alpha/models/{model_name}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [{
            "parts": [
                {"text": "Extract structured data from this document."},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded_string}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    time_taken = time.time() - start_time
    response.raise_for_status()
    result = response.json()
    
    text_content = result['candidates'][0]['content']['parts'][0]['text']
    usage = result.get('usageMetadata', {})
    stats = {
        "model_used": model_name,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "time_taken": round(time_taken, 2),
        "estimated_cost": 0.0005
    }
    return text_content, stats

def call_vision_model(image_path: str, system_prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Calls OpenRouter with gpt-4o-mini, falls back to Google Gemini."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    vision_primary = os.getenv("VISION_PRIMARY_MODEL", "openai/gpt-4o-mini")
    vision_fallback = os.getenv("VISION_FALLBACK_MODEL", "gemini-3-flash-preview")
    
    # Attempt OpenRouter (vision primary)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            logger.info(f"Calling vision model: {vision_primary} via OpenRouter")
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": vision_primary,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract structured data from this document."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                    ]}
                ],
                "response_format": {"type": "json_object"}
            }
            start_time = time.time()
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            time_taken = time.time() - start_time
            response.raise_for_status()
            result = response.json()
            
            text_content = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            stats = {
                "model_used": vision_primary,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "time_taken": round(time_taken, 2),
                "estimated_cost": 0.0001
            }
            return text_content, stats
        except Exception as e:
            logger.warning(f"{vision_primary} vision call failed: {e}. Falling back to {vision_fallback}.")
    
    # Fallback to Google Gemini
    return call_google_gemini_vision(image_path, system_prompt, model_name=vision_fallback)

def extract_json_with_llm(system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attempts to extract JSON using the primary model. 
    If it fails or returns invalid JSON, falls back to the VLM (Gemini Flash).
    """
    primary_model = os.getenv("PRIMARY_MODEL", "gpt-oss-120b")
    fallback_model = os.getenv("FALLBACK_MODEL", "meta-llama/llama-3.1-70b-instruct")
    
    # Try primary
    try:
        logger.info(f"Calling primary model: {primary_model}")
        text_content, stats = call_openrouter(primary_model, system_prompt, user_prompt)
        # Parse JSON
        parsed = json.loads(text_content)
        return parsed, stats
    except Exception as e:
        logger.warning(f"Primary model {primary_model} failed: {e}. Falling back to {fallback_model}.")
        
        # Try fallback
        try:
            logger.info(f"Calling fallback model via OpenRouter: {fallback_model}")
            text_content, stats = call_openrouter(fallback_model, system_prompt, user_prompt)
            parsed = json.loads(text_content)
            return parsed, stats
        except Exception as fallback_e:
            logger.error(f"Fallback model also failed: {fallback_e}")
            raise ValueError(f"Both primary and fallback models failed. Last error: {fallback_e}")


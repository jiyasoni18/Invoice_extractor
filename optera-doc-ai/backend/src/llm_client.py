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

def call_gemini_vision(image_path: str, system_prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Calls the native Google Gemini API with an image directly (VLM Baseline)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
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
        "model_used": "gemini-1.5-flash-vision",
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "time_taken": round(time_taken, 2),
        "estimated_cost": 0.0005 # Baseline vision models cost more natively
    }
    
    return text_content, stats

def extract_json_with_llm(system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attempts to extract JSON using the primary model. 
    If it fails or returns invalid JSON, falls back to the VLM (Gemini Flash).
    """
    primary_model = os.getenv("PRIMARY_MODEL", "gpt-oss-120b")
    fallback_model = os.getenv("FALLBACK_MODEL", "gemini-1.5-flash")
    
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
            logger.info(f"Calling fallback model natively via Google API: {fallback_model}")
            text_content, stats = call_google_gemini(fallback_model, system_prompt, user_prompt)
            parsed = json.loads(text_content)
            return parsed, stats
        except Exception as fallback_e:
            logger.error(f"Fallback model also failed: {fallback_e}")
            raise ValueError(f"Both primary and fallback models failed. Last error: {fallback_e}")


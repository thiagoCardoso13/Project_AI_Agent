"""Fábrica de modelos: o provedor é escolhido por variável de ambiente."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # lê o arquivo .env da pasta do projeto, se existir

DEFAULTS = {
    "gemini": "gemini-3.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "ollama": "qwen2.5:14b",
}


def get_setting(name: str, default: str | None = None) -> str | None:
    """Lê de variável de ambiente e, se existir, dos secrets do Streamlit."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


# Modelos de reserva: cada um tem cota gratuita própria, então se um estourar o limite o agente usa o próximo.
GEMINI_FALLBACKS = ["gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]


def get_fallback_llms() -> list:
    """Modelos alternativos (mesmo provedor) usados quando o principal falha, ex.: cota esgotada."""
    provider = (get_setting("LLM_PROVIDER", "gemini") or "gemini").lower()
    if provider != "gemini":
        return []
    raw = get_setting("LLM_FALLBACK_MODELS")
    names = [n.strip() for n in raw.split(",")] if raw else GEMINI_FALLBACKS
    return [get_llm(model=n) for n in names if n]


def get_llm(model: str | None = None):
    provider = (get_setting("LLM_PROVIDER", "gemini") or "gemini").lower()
    model = model or get_setting("LLM_MODEL", DEFAULTS.get(provider))

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_setting("GOOGLE_API_KEY"),
            temperature=0,
            timeout=60,
            max_retries=2,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=get_setting("GROQ_API_KEY"), temperature=0)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=get_setting("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)
    raise ValueError(f"LLM_PROVIDER desconhecido: {provider}")

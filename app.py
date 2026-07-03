#!/usr/bin/env python3
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Tuple

DEFAULT_MESSAGE = "I have sudden chest pain and I am having trouble breathing."


def build_prompt_variant_1(patient_message: str) -> str:
    return f"""You are an AfyaPlus triage assistant.
Analyze the patient message and return a JSON object with these keys:
- is_critical_emergency (boolean)
- detected_symptoms (array of strings)
- clinical_reasoning_summary (string)
- routing_destination (string)
Patient message: {patient_message}"""


def build_prompt_variant_2(patient_message: str) -> str:
    return f"""You are an AfyaPlus triage assistant. Use a conservative medical safety posture.
Classify the message as emergency or routine, list symptoms, provide a concise reasoning summary, and choose a routing destination.
Return only valid JSON matching this schema:
{{"is_critical_emergency": boolean, "detected_symptoms": ["string"], "clinical_reasoning_summary": "string", "routing_destination": "string"}}
Patient message: {patient_message}"""


def build_prompt_variant_3(patient_message: str) -> str:
    return f"""You are AfyaPlus Triage Officer, a safety-focused clinical intake classifier.
Your job is to process patient messages into a strict machine-readable triage record.

Follow this reasoning path before answering:
1. Extract the main symptoms explicitly mentioned.
2. Decide whether the case looks like a critical emergency or a routine inquiry.
3. Write a short, evidence-based reasoning summary.
4. Choose the most appropriate routing destination.

Guardrails:
- Do not add conversational fluff, greetings, or disclaimers.
- Do not invent symptoms or clinical facts not present in the message.
- Do not perform medical calculations or claim certainty beyond the text.
- Return only a single JSON object and nothing else.
- Output must match this schema exactly:
{{"is_critical_emergency": boolean, "detected_symptoms": ["string", "string"], "clinical_reasoning_summary": "string", "routing_destination": "string"}}

Patient message: {patient_message}"""


def normalize_json_response(raw_response: object) -> Dict[str, object]:
    if isinstance(raw_response, dict):
        return raw_response

    if isinstance(raw_response, str):
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("The model response was not valid JSON")


def ensure_schema(payload: Dict[str, object]) -> Dict[str, object]:
    normalized = {
        "is_critical_emergency": bool(payload.get("is_critical_emergency", False)),
        "detected_symptoms": [
            str(item)
            for item in payload.get("detected_symptoms", [])
            if str(item).strip()
        ],
        "clinical_reasoning_summary": str(payload.get("clinical_reasoning_summary", "No reasoning provided.")).strip(),
        "routing_destination": str(payload.get("routing_destination", "general_inquiry")).strip() or "general_inquiry",
    }
    if not normalized["detected_symptoms"]:
        normalized["detected_symptoms"] = ["general concern"]
    return normalized


def call_cloud_model(patient_message: str, prompt: str, timeout: float = 4.0) -> Tuple[Dict[str, object], float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": patient_message},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    elapsed = time.perf_counter() - start

    data = json.loads(body)
    content = data["choices"][0]["message"]["content"]
    return ensure_schema(normalize_json_response(content)), elapsed


def call_local_ollama(patient_message: str, prompt: str, timeout: float = 4.0) -> Tuple[Dict[str, object], float]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": patient_message},
        ],
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    elapsed = time.perf_counter() - start

    data = json.loads(body)
    content = data.get("message", {}).get("content", "")
    return ensure_schema(normalize_json_response(content)), elapsed


def build_rule_based_fallback(patient_message: str) -> Dict[str, object]:
    message = patient_message.lower()
    symptoms = []
    critical_keywords = [
        "chest pain",
        "shortness of breath",
        "trouble breathing",
        "severe bleeding",
        "faint",
        "unconscious",
        "seizure",
        "stroke",
        "vision loss",
        "severe headache",
        "convulsions",
        "blood pressure",
    ]
    routine_keywords = ["cold", "flu", "cough", "rash", "headache", "sore throat", "allergy"]

    if any(keyword in message for keyword in ["pain", "bleeding", "breath", "seizure", "stroke", "unconscious"]):
        symptoms.append("acute discomfort")
    if any(keyword in message for keyword in ["chest", "breath", "shortness", "difficulty"]):
        symptoms.append("respiratory distress")
    if any(keyword in message for keyword in ["fever", "cough", "sore", "rash"]):
        symptoms.append("mild infection symptoms")

    is_critical = any(keyword in message for keyword in critical_keywords)
    routing_destination = "emergency_room" if is_critical else "telehealth"
    if not is_critical and any(keyword in message for keyword in routine_keywords):
        routing_destination = "urgent_care"

    summary = (
        "The message contains signs of urgent care needs and was handled with a safe offline fallback."
        if not is_critical
        else "The message includes high-risk symptoms that warrant immediate emergency evaluation."
    )
    return {
        "is_critical_emergency": is_critical,
        "detected_symptoms": symptoms or ["general concern"],
        "clinical_reasoning_summary": summary,
        "routing_destination": routing_destination,
    }


def triage_patient(patient_message: str) -> Tuple[Dict[str, object], str, float, float]:
    prompt = build_prompt_variant_3(patient_message)

    cloud_result = None
    cloud_latency = None
    try:
        cloud_result, cloud_latency = call_cloud_model(patient_message, prompt, timeout=4.0)
        return cloud_result, "cloud", cloud_latency, 0.0
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Cloud request failed: {exc}. Falling back to local Ollama.", file=sys.stderr)

    try:
        local_result, local_latency = call_local_ollama(patient_message, prompt, timeout=4.0)
        return local_result, "ollama", cloud_latency or 0.0, local_latency
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, ValueError, json.JSONDecodeError) as exc:
        print(f"Local Ollama request failed: {exc}. Using rule-based fallback.", file=sys.stderr)

    fallback_result = build_rule_based_fallback(patient_message)
    return fallback_result, "rule_based_fallback", cloud_latency or 0.0, 0.0


def main() -> None:
    patient_message = " ".join(sys.argv[1:]).strip() or DEFAULT_MESSAGE
    result, provider, cloud_latency, local_latency = triage_patient(patient_message)
    print(json.dumps(result, indent=2))
    print(f"Routing decision: {result['routing_destination']} via {provider}")
    print(f"Cloud latency: {cloud_latency:.3f}s | Local latency: {local_latency:.3f}s")


if __name__ == "__main__":
    main()

import json
import requests


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "llama3.2:latest"


def build_judge_prompt(
    question: str,
    expected_answer: str,
    generated_answer: str,
) -> str:
    return f"""
Tu es un évaluateur strict d’un assistant IA pour dashboard.

Tu dois comparer la réponse générée à la réponse attendue.

Règles :
- Accepte une formulation différente si le sens métier est correct.
- Refuse une réponse qui invente des données.
- Refuse une réponse qui donne une valeur métier incorrecte.
- Refuse une réponse hors contexte dashboard.
- Pour les questions adverses, une bonne réponse peut être un refus clair d’inventer.
- Réponds uniquement en JSON valide.

Question :
{question}

Réponse attendue :
{expected_answer}

Réponse générée :
{generated_answer}

Format attendu :
{{
  "decision": "PASS" ou "FAIL",
  "score": 0 à 5,
  "reason": "justification courte"
}}
""".strip()


def judge_with_ollama(
    question: str,
    expected_answer: str,
    generated_answer: str,
    timeout_seconds: int = 60,
) -> dict:
    prompt = build_judge_prompt(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un juge d’évaluation IA. Tu réponds uniquement en JSON valide.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 160,
        },
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        content = data.get("message", {}).get("content", "").strip()

        return parse_judge_json(content)

    except Exception as error:
        return {
            "decision": "FAIL",
            "score": 0,
            "reason": f"Erreur LLM judge : {error}",
        }


def parse_judge_json(content: str) -> dict:
    try:
        return json.loads(content)

    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "decision": "FAIL",
        "score": 0,
        "reason": "Réponse du juge non parseable en JSON.",
    }
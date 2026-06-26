import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character for character in text
        if unicodedata.category(character) != "Mn"
    )
    text = text.replace("€", " euros ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_all_terms(answer: str, required_terms: list[str]) -> tuple[bool, list[str]]:
    normalized_answer = normalize_text(answer)
    missing_terms = []

    for term in required_terms:
        normalized_term = normalize_text(term)

        if normalized_term not in normalized_answer:
            missing_terms.append(term)

    return len(missing_terms) == 0, missing_terms


def contains_forbidden_terms(answer: str, forbidden_terms: list[str]) -> tuple[bool, list[str]]:
    normalized_answer = normalize_text(answer)
    found_terms = []

    for term in forbidden_terms:
        normalized_term = normalize_text(term)

        if normalized_term and normalized_term in normalized_answer:
            found_terms.append(term)

    return len(found_terms) > 0, found_terms


def check_short_answer(answer: str, max_words: int = 60) -> bool:
    if not answer:
        return False

    return len(answer.split()) <= max_words


def check_not_empty(answer: str) -> bool:
    return bool(answer and answer.strip())


def compute_deterministic_metrics(
    test_case: dict[str, Any],
    generated_answer: str,
) -> dict[str, Any]:
    must_contain = test_case.get("must_contain", [])
    must_not_contain = test_case.get("must_not_contain", [])

    contains_required, missing_terms = contains_all_terms(
        generated_answer,
        must_contain,
    )

    has_forbidden, forbidden_found = contains_forbidden_terms(
        generated_answer,
        must_not_contain,
    )

    answer_not_empty = check_not_empty(generated_answer)
    answer_is_short = check_short_answer(generated_answer)

    string_matching_pass = contains_required and not has_forbidden
    deterministic_pass = (
        answer_not_empty
        and contains_required
        and not has_forbidden
        and answer_is_short
    )

    score = 0

    if answer_not_empty:
        score += 1

    if contains_required:
        score += 2

    if not has_forbidden:
        score += 1

    if answer_is_short:
        score += 1

    return {
        "deterministic_pass": deterministic_pass,
        "string_matching_pass": string_matching_pass,
        "score": score,
        "score_max": 5,
        "answer_not_empty": answer_not_empty,
        "answer_is_short": answer_is_short,
        "contains_required": contains_required,
        "missing_terms": missing_terms,
        "has_forbidden_terms": has_forbidden,
        "forbidden_terms_found": forbidden_found,
    }


def compute_final_decision(
    deterministic_metrics: dict[str, Any],
    judge_result: dict[str, Any] | None,
) -> str:
    if judge_result is None:
        return "PASS" if deterministic_metrics["deterministic_pass"] else "FAIL"

    judge_decision = judge_result.get("decision", "FAIL")
    judge_score = int(judge_result.get("score", 0))

    if deterministic_metrics["deterministic_pass"] and judge_decision == "PASS":
        return "PASS"

    if deterministic_metrics["string_matching_pass"] and judge_score >= 4:
        return "PASS"

    return "FAIL"
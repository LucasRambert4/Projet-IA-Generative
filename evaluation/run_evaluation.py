import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
DATASET_PATH = EVALUATION_DIR / "datasets" / "dashboard_eval_dataset.json"
RESULTS_DIR = EVALUATION_DIR / "results"

# Important:
# When this file is launched with:
# python evaluation/run_evaluation.py
# Python starts from the evaluation folder.
# So we manually add the project root to import src/.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.ollama_client import ask_ollama
from src.dashboard_ai_engine import (
    answer_from_dashboard_facts,
    get_grounded_dashboard_context,
)
from evaluation.metrics import compute_deterministic_metrics, compute_final_decision
from evaluation.llm_judge import judge_with_ollama


DASHBOARD_CONTEXT = get_grounded_dashboard_context()

def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_answer(question: str) -> str:
    """
    Generates an answer using the same grounded approach as the assistant.

    Priority:
    1. deterministic dashboard facts
    2. controlled refusal for out-of-scope questions
    3. Ollama only when the answer cannot be resolved directly
    """

    grounded_answer = answer_from_dashboard_facts(question)

    if grounded_answer:
        return grounded_answer

    return ask_ollama(
        user_message=question,
        conversation_history=[],
        dashboard_context=DASHBOARD_CONTEXT,
    )

def evaluate_test_case(
    test_case: dict[str, Any],
    use_llm_judge: bool,
) -> dict[str, Any]:
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    start_time = time.time()

    try:
        generated_answer = generate_answer(question)
        generation_error = None
    except Exception as error:
        generated_answer = ""
        generation_error = str(error)

    latency_seconds = round(time.time() - start_time, 3)

    deterministic_metrics = compute_deterministic_metrics(
        test_case=test_case,
        generated_answer=generated_answer,
    )

    judge_result = None

    if use_llm_judge and generated_answer:
        judge_result = judge_with_ollama(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )

    final_decision = compute_final_decision(
        deterministic_metrics=deterministic_metrics,
        judge_result=judge_result,
    )

    return {
        "id": test_case["id"],
        "scenario": test_case["scenario"],
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "latency_seconds": latency_seconds,
        "generation_error": generation_error,
        "llm_as_judge": judge_result,
        "string_matching": deterministic_metrics["string_matching_pass"],
        "deterministic_pass": deterministic_metrics["deterministic_pass"],
        "check_answer_not_empty": deterministic_metrics["answer_not_empty"],
        "check_answer_is_short": deterministic_metrics["answer_is_short"],
        "check_contains_required": deterministic_metrics["contains_required"],
        "check_missing_terms": deterministic_metrics["missing_terms"],
        "check_has_forbidden_terms": deterministic_metrics["has_forbidden_terms"],
        "check_forbidden_terms_found": deterministic_metrics["forbidden_terms_found"],
        "deterministic_score": deterministic_metrics["score"],
        "deterministic_score_max": deterministic_metrics["score_max"],
        "final_decision": final_decision,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result["final_decision"] == "PASS")
    failed = total - passed

    latencies = [
        result["latency_seconds"]
        for result in results
        if result.get("latency_seconds") is not None
    ]

    summary_by_scenario = {}

    for result in results:
        scenario = result["scenario"]

        if scenario not in summary_by_scenario:
            summary_by_scenario[scenario] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "success_rate": 0.0,
            }

        summary_by_scenario[scenario]["total"] += 1

        if result["final_decision"] == "PASS":
            summary_by_scenario[scenario]["passed"] += 1
        else:
            summary_by_scenario[scenario]["failed"] += 1

    for scenario, scenario_summary in summary_by_scenario.items():
        scenario_summary["success_rate"] = round(
            scenario_summary["passed"] / scenario_summary["total"] * 100,
            2,
        )

    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "success_rate": round(passed / total * 100, 2) if total else 0.0,
        "average_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
        "summary_by_scenario": summary_by_scenario,
    }


def save_json_results(results: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "evaluation_results.json"

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "simulate_inputs_generate_outputs_evaluate_outputs",
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return output_path


def save_csv_results(results: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "evaluation_results.csv"

    fieldnames = [
        "id",
        "scenario",
        "question",
        "expected_answer",
        "generated_answer",
        "latency_seconds",
        "final_decision",
        "string_matching",
        "deterministic_pass",
        "check_answer_not_empty",
        "check_answer_is_short",
        "check_contains_required",
        "check_missing_terms",
        "check_has_forbidden_terms",
        "check_forbidden_terms_found",
        "deterministic_score",
        "deterministic_score_max",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                key: result.get(key)
                for key in fieldnames
            })

    return output_path


def print_summary(summary: dict[str, Any]):
    print()
    print("========== RÉSUMÉ ÉVALUATION ==========")
    print(f"Nombre total de tests : {summary['total_tests']}")
    print(f"Tests réussis : {summary['passed']}")
    print(f"Tests échoués : {summary['failed']}")
    print(f"Score global : {summary['success_rate']} %")
    print(f"Latence moyenne : {summary['average_latency_seconds']} s")
    print(f"Latence max : {summary['max_latency_seconds']} s")

    print()
    print("Score par scénario :")

    for scenario, scenario_summary in summary["summary_by_scenario"].items():
        print(
            f"- {scenario} : "
            f"{scenario_summary['passed']}/{scenario_summary['total']} "
            f"({scenario_summary['success_rate']} %)"
        )

    print("=======================================")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline d’évaluation du POC IA générative."
    )

    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Désactive le LLM as a judge.",
    )

    args = parser.parse_args()

    use_llm_judge = not args.no_judge

    dataset = load_dataset(DATASET_PATH)

    results = []

    print("Lancement du pipeline d’évaluation...")
    print("Étape 1 : simulation des inputs")
    print("Étape 2 : génération des outputs")
    print("Étape 3 : évaluation des outputs")
    print()

    for index, test_case in enumerate(dataset, start=1):
        print(
            f"[{index}/{len(dataset)}] "
            f"{test_case['id']} | scénario={test_case['scenario']}"
        )

        result = evaluate_test_case(
            test_case=test_case,
            use_llm_judge=use_llm_judge,
        )

        results.append(result)

        print(f"Question : {result['question']}")
        print(f"Réponse générée : {result['generated_answer']}")
        print(f"Décision finale : {result['final_decision']}")
        print()

    summary = summarize_results(results)

    json_path = save_json_results(
        results=results,
        summary=summary,
    )

    csv_path = save_csv_results(results)

    print_summary(summary)

    print(f"Résultats JSON : {json_path}")
    print(f"Résultats CSV : {csv_path}")


if __name__ == "__main__":
    main()
from typer.testing import CliRunner

from app.evals import run_evals
from app.evals.runner import MockEvalToolSuite
from exec_agent.cli import app


def test_mock_eval_harness_passes_all_tasks() -> None:
    results = run_evals(MockEvalToolSuite())

    assert len(results) == 6
    assert {result.category for result in results} == {
        "memory retrieval",
        "PDF QA",
        "DOCX QA",
        "web research",
        "image description",
        "HITL approval flow",
    }
    assert all(result.passed for result in results)


def test_eval_run_cli_outputs_rich_results_table() -> None:
    result = CliRunner().invoke(app, ["eval", "run"])

    assert result.exit_code == 0
    assert "Executive Assistant Eval Results" in result.output
    assert "memory_retrieval_budget_context" in result.output
    assert "pdf_qa_receipt_threshold" in result.output
    assert "docx_qa_approval_owner" in result.output
    assert "web_research_source_summary" in result.output
    assert "image_description_chart" in result.output
    assert "hitl_approval_allows_safe_action" in result.output
    assert "PASS" in result.output

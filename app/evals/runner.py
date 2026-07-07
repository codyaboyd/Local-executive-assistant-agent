"""Offline evaluation harness with mocked tool implementations.

The default harness intentionally avoids network, GPU, and heavyweight model calls so
it can run in CI. Each sample task exercises one assistant capability through a
small deterministic fixture and pass/fail assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich.table import Table


@dataclass(frozen=True)
class EvalResult:
    """Result for one evaluation task."""

    name: str
    category: str
    passed: bool
    details: str


class EvalToolSuite(Protocol):
    """Tool interface used by eval tasks.

    Implementations may call real tools for manual/local experiments, while the
    default mock implementation is deterministic and safe for CI.
    """

    def search_memory(self, query: str) -> list[str]: ...

    def answer_pdf_question(self, question: str) -> str: ...

    def answer_docx_question(self, question: str) -> str: ...

    def search_web(self, query: str) -> list[dict[str, str]]: ...

    def describe_image(self, image_path: str) -> str: ...

    def request_approval(self, action: str, preview: str) -> bool: ...


@dataclass
class MockEvalToolSuite:
    """Deterministic mocked tools for CI-safe eval runs."""

    approval_response: bool = True

    def search_memory(self, query: str) -> list[str]:
        memories = [
            "User prefers Monday morning travel briefings.",
            "CFO asks for budget deltas before board meetings.",
        ]
        query_terms = {term.lower().strip(".,?!") for term in query.split()}
        return [memory for memory in memories if query_terms & {term.lower().strip(".,?!") for term in memory.split()}]

    def answer_pdf_question(self, question: str) -> str:
        return "The PDF travel policy says receipts are required for expenses over $25. Source: travel-policy.pdf p. 2."

    def answer_docx_question(self, question: str) -> str:
        return "The DOCX onboarding guide names Priya as the approval owner. Source: onboarding.docx section: Approvals."

    def search_web(self, query: str) -> list[dict[str, str]]:
        return [
            {
                "title": "Example market update",
                "url": "https://example.test/market-update",
                "snippet": "Acme revenue grew 12% year over year in the mocked research fixture.",
            }
        ]

    def describe_image(self, image_path: str) -> str:
        return "A bar chart showing Q1 revenue rising from January to March."

    def request_approval(self, action: str, preview: str) -> bool:
        return self.approval_response


def run_evals(tools: EvalToolSuite | None = None) -> list[EvalResult]:
    """Run all bundled sample eval tasks and return pass/fail results."""

    suite = tools or MockEvalToolSuite()
    return [
        _eval_memory_retrieval(suite),
        _eval_pdf_qa(suite),
        _eval_docx_qa(suite),
        _eval_web_research(suite),
        _eval_image_description(suite),
        _eval_hitl_approval_flow(suite),
    ]


def render_results_table(results: list[EvalResult]) -> Table:
    """Render eval results as a Rich table."""

    table = Table(title="Executive Assistant Eval Results")
    table.add_column("Status", no_wrap=True)
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta", no_wrap=True)
    table.add_column("Details", style="white")
    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        table.add_row(status, result.name, result.category, result.details)
    return table


def _eval_memory_retrieval(tools: EvalToolSuite) -> EvalResult:
    memories = tools.search_memory("budget board meeting")
    passed = any("budget" in memory.lower() and "board" in memory.lower() for memory in memories)
    return EvalResult(
        name="memory_retrieval_budget_context",
        category="memory retrieval",
        passed=passed,
        details="Retrieved CFO budget memory." if passed else "Expected a budget/board memory result.",
    )


def _eval_pdf_qa(tools: EvalToolSuite) -> EvalResult:
    answer = tools.answer_pdf_question("What receipt threshold is in the travel policy PDF?")
    passed = "$25" in answer and "travel-policy.pdf" in answer and "p. 2" in answer
    return EvalResult(
        name="pdf_qa_receipt_threshold",
        category="PDF QA",
        passed=passed,
        details="Answered with threshold and PDF page citation." if passed else f"Unexpected answer: {answer}",
    )


def _eval_docx_qa(tools: EvalToolSuite) -> EvalResult:
    answer = tools.answer_docx_question("Who owns onboarding approval in the DOCX?")
    passed = "priya" in answer.lower() and "onboarding.docx" in answer and "Approvals" in answer
    return EvalResult(
        name="docx_qa_approval_owner",
        category="DOCX QA",
        passed=passed,
        details="Answered with owner and DOCX section citation." if passed else f"Unexpected answer: {answer}",
    )


def _eval_web_research(tools: EvalToolSuite) -> EvalResult:
    results = tools.search_web("Acme revenue growth")
    passed = bool(results) and results[0].get("url", "").startswith("https://") and "12%" in results[0].get("snippet", "")
    return EvalResult(
        name="web_research_source_summary",
        category="web research",
        passed=passed,
        details="Returned mocked web source with expected growth fact." if passed else "Expected mocked HTTPS source and 12% fact.",
    )


def _eval_image_description(tools: EvalToolSuite) -> EvalResult:
    description = tools.describe_image("fixtures/revenue-chart.png")
    passed = "bar chart" in description.lower() and "revenue" in description.lower()
    return EvalResult(
        name="image_description_chart",
        category="image description",
        passed=passed,
        details="Described chart content without GPU inference." if passed else f"Unexpected description: {description}",
    )


def _eval_hitl_approval_flow(tools: EvalToolSuite) -> EvalResult:
    approved = tools.request_approval("crawl", "Crawl example.test with max 2 pages")
    return EvalResult(
        name="hitl_approval_allows_safe_action",
        category="HITL approval flow",
        passed=approved,
        details="Mock approval allowed the guarded action." if approved else "Expected mocked approval to allow the action.",
    )

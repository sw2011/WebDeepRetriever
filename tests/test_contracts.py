from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from web_agent.contracts import ActionReceipt, CoverageCertificate, TaskContract
from web_agent.evidence import EvidenceStore
from web_agent.verifier import CompletionVerifier


def contract(task: str = "查询页面中的值", *, requires_form_confirmation: bool = False) -> TaskContract:
    return TaskContract.from_item(
        {
            "task_idx": 1,
            "task_id": "task-1",
            "website": "https://example.test",
            "task": task,
            "requires_form_confirmation": requires_form_confirmation,
        }
    )


def successful_receipt(confirmation: bool = False) -> ActionReceipt:
    return ActionReceipt(
        "act-1",
        "click",
        True,
        "https://example.test",
        "https://example.test/done",
        "before",
        "after",
        {"confirmation": confirmation},
    )


def test_contract_caps_protocol_steps_and_infers_coverage_only() -> None:
    value = TaskContract.from_item(
        {
            "task_idx": 3,
            "task_id": "x",
            "website": "example.test",
            "task": "列出所有记录并提交申请",
            "requires_form_confirmation": True,
        },
        max_steps=999,
    )
    assert value.website == "https://example.test"
    assert value.max_steps == 100
    assert value.requires_coverage is True
    assert value.requires_form_confirmation is True

    english = TaskContract.from_item(
        {"task_idx": 2, "task_id": "e", "website": "example.test", "task": "How many records are there?"}
    )
    assert english.requires_coverage is True

    assert contract("Which year had the highest revenue?").requires_coverage is False
    assert contract("Which companies are included in the index?").requires_coverage is True
    assert contract("Indicator=Top 10% income share, Year=2010; what is the value (%)?").requires_coverage is False
    assert contract("北京和上海的人口分别是多少？").requires_coverage is False
    assert contract("查询公司的所有权结构").requires_coverage is False
    assert contract("查询公司的完整名称").requires_coverage is False
    assert contract("查询该大学的世界排名").requires_coverage is False
    assert contract("查询公司的完整信息").requires_coverage is False
    assert contract("查询软件的完整版本号").requires_coverage is False
    assert contract("查询页面的完整 URL").requires_coverage is False
    assert contract("查询该企业的所有制性质").requires_coverage is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (None, False),
    ],
)
def test_form_confirmation_requires_explicit_boolean_true(value: object, expected: bool) -> None:
    item = {
        "task_idx": 1,
        "task_id": "form",
        "website": "example.test",
        "task": "填写并提交申请",
    }
    if value is not None:
        item["requires_form_confirmation"] = value
    assert TaskContract.from_item(item).requires_form_confirmation is expected


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_invalid_explicit_form_confirmation_type_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="JSON 布尔值"):
        TaskContract.from_item(
            {
                "task_idx": 1,
                "task_id": "form",
                "website": "example.test",
                "task": "填写并提交申请",
                "requires_form_confirmation": value,
            }
        )


@pytest.mark.parametrize(
    "task",
    [
        "查询公司首次提交 10-K 年报的日期",
        "查询专利申请号对应的申请日期",
        "查找机构发布的年度报告",
        "检索已发送公告的发布时间",
    ],
)
def test_retrieval_wording_never_implies_form_side_effect(task: str) -> None:
    assert contract(task).requires_form_confirmation is False


def test_fixed_sample_task_37_is_retrieval_not_form_submission() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "protocol3_sample_8.json"
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    item = next(value for value in items if value["task_idx"] == 37)
    value = TaskContract.from_item(item)
    assert value.requires_form_confirmation is False
    assert value.requires_coverage is False


def test_fixed_sample_task_25_requires_collection_coverage() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "protocol3_sample_8.json"
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    item = next(value for value in items if value["task_idx"] == 25)
    value = TaskContract.from_item(item)
    assert value.requires_coverage is True
    assert contract("查询排名第几个").requires_coverage is False


def test_verifier_rejects_empty_unknown_and_unvisited_evidence() -> None:
    store = EvidenceStore()
    evidence = store.add("dom", "https://other.test", "fact", {"value": "42"})
    result = CompletionVerifier().verify(
        contract(),
        "42",
        [evidence.evidence_id, "ev-missing"],
        {"$": [evidence.evidence_id]},
        None,
        store,
        [],
        ["https://example.test"],
    )
    assert result.accepted is False
    assert any("不存在" in reason for reason in result.reasons)
    assert any("URL 未被访问" in reason for reason in result.reasons)


def test_verifier_requires_every_answer_leaf_binding() -> None:
    store = EvidenceStore()
    evidence = store.add("dom", "https://example.test", "row", {"data": ["A", "B"]})
    result = CompletionVerifier().verify(
        contract(),
        {"name": "A", "value": "B"},
        [evidence.evidence_id],
        {"$.name": [evidence.evidence_id]},
        None,
        store,
        [],
        ["https://example.test"],
    )
    assert result.accepted is False
    assert "答案字段 $.value 未绑定证据" in result.reasons


def test_verifier_accepts_evidence_backed_scalar_answer() -> None:
    store = EvidenceStore()
    evidence = store.add("dom", "https://example.test", "answer", {"value": "42"})
    result = CompletionVerifier().verify(
        contract(),
        "42",
        [evidence.evidence_id],
        {"$": [evidence.evidence_id]},
        None,
        store,
        [],
        ["https://example.test"],
    )
    assert result.accepted is True


def test_form_task_needs_positive_confirmation_receipt() -> None:
    store = EvidenceStore()
    evidence = store.add("dom", "https://example.test/done", "submitted", {"value": "ok"})
    verifier = CompletionVerifier()
    args = (
        contract("填写并提交申请", requires_form_confirmation=True),
        "提交成功",
        [evidence.evidence_id],
        {"$": [evidence.evidence_id]},
        None,
        store,
    )
    rejected = verifier.verify(*args, [successful_receipt(False)], ["https://example.test/done"])
    accepted = verifier.verify(*args, [successful_receipt(True)], ["https://example.test/done"])
    assert rejected.accepted is False
    assert accepted.accepted is True


def test_coverage_requires_terminal_evidence_counts_and_fingerprint() -> None:
    store = EvidenceStore()
    answer_evidence = store.add("dom", "https://example.test", "rows", {"data": ["A", "B"]})
    terminal = store.add("dom", "https://example.test", "next disabled", {"disabled": True})
    certificate = CoverageCertificate(
        strategy="pagination",
        unique_item_count=2,
        pages_visited=2,
        expected_total=2,
        terminal_reason="next_disabled",
        terminal_evidence_id=terminal.evidence_id,
        item_fingerprint="a" * 64,
    )
    verifier = CompletionVerifier()
    result = verifier.verify(
        contract("列出所有记录"),
        "A, B",
        [answer_evidence.evidence_id],
        {"$": [answer_evidence.evidence_id]},
        certificate,
        store,
        [],
        ["https://example.test"],
    )
    assert result.accepted is True
    mismatch = verifier.verify(
        contract("列出所有记录"),
        "A, B",
        [answer_evidence.evidence_id],
        {"$": [answer_evidence.evidence_id]},
        replace(certificate, expected_total=3),
        store,
        [],
        ["https://example.test"],
    )
    assert mismatch.accepted is False
    assert any("不一致" in reason for reason in mismatch.reasons)


def test_evidence_store_is_append_only_and_ids_are_deterministic(tmp_path) -> None:
    store = EvidenceStore()
    first = store.add("dom", "https://example.test", "one", {"x": 1})
    second = store.add(
        "network",
        "https://user:password@example.test?access_token=private",
        "two",
        {"message": "Bearer private", "session": "private"},
    )
    assert (first.evidence_id, second.evidence_id) == ("ev-00001", "ev-00002")
    assert "private" not in str(second.to_dict())
    assert "password" not in second.url
    with pytest.raises(Exception):
        first.payload = {"changed": True}  # type: ignore[misc]
    path = tmp_path / "evidence.json"
    store.save(path)
    serialized = path.read_text(encoding="utf-8")
    assert '"ev-00002"' in serialized
    assert "private" not in serialized

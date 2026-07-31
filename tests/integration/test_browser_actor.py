from __future__ import annotations

import asyncio
import gzip
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from agents.tool_context import ToolContext
from openai import AsyncOpenAI
from PIL import Image

from web_agent.browser_actor import BrowserActor
from web_agent.contracts import TaskContract
from web_agent.runtime import TaskRuntimeContext, observe as observe_tool
from web_agent.verifier import CompletionVerifier

from conftest import one_element


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_large_dom_model_projection_repeat_and_full_audit(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/large-dom")
    context = TaskRuntimeContext(
        actor=actor,
        contract=TaskContract.from_item(
            {
                "task_idx": 1,
                "task_id": "large",
                "website": "https://example.test",
                "task": "Download critical metric",
            }
        ),
        evidence_store=actor.evidence_store,
        verifier=CompletionVerifier(),
        vision_client=AsyncOpenAI(api_key="test", base_url="http://127.0.0.1:9/v1", max_retries=0),
        vision_model="test",
    )

    first_text = await observe_tool.on_invoke_tool(
        ToolContext(context, tool_name="observe", tool_call_id="1", tool_arguments="{}"),
        "{}",
    )
    first = json.loads(first_text)
    assert first["total_element_count"] >= 1_200
    assert first["element_count"] <= 120
    assert len(first_text) <= 24_000
    critical = next(item for item in first["elements"] if item.get("text") == "Download critical metric")

    second_text = await observe_tool.on_invoke_tool(
        ToolContext(context, tool_name="observe", tool_call_id="2", tool_arguments="{}"),
        "{}",
    )
    second = json.loads(second_text)
    assert second["unchanged"] is True
    assert second["bids_remain_valid"] is True
    assert len(second["elements"]) <= 24
    assert any(item["bid"] == critical["bid"] for item in second["elements"])
    assert (await actor.click(critical["bid"]))["success"] is True

    artifacts = sorted((actor.output_dir / "observations").glob("*.json.gz"))
    assert len(artifacts) == 2
    with gzip.open(artifacts[-1], "rt", encoding="utf-8") as source:
        audit = json.load(source)
    assert len(audit["elements"]) == second["total_element_count"]
    assert audit["dom_hash"] == second["dom_hash"]
    assert "dom_snapshot" in audit and "ax_tree" in audit


async def test_native_custom_form_duplicate_labels_stale_bid_and_spa(
    actor_factory: Any,
) -> None:
    actor: BrowserActor = await actor_factory("/forms")
    observation = await actor.observe()

    contact_inputs = [
        element
        for element in observation["elements"]
        if element.get("tag") == "input" and element.get("label") == "Contact"
    ]
    assert len(contact_inputs) == 2
    assert len({element["bid"] for element in contact_inputs}) == 2
    primary = one_element(observation, tag="input", name="primary")
    secondary = one_element(observation, tag="input", name="secondary")
    country = one_element(observation, tag="select", name="country")
    terms = one_element(observation, tag="input", name="terms")
    volatile = one_element(observation, tag="input", name="volatile")

    primary_receipt = await actor.fill(primary["bid"], "primary@example.test")
    secondary_receipt = await actor.fill(secondary["bid"], "secondary@example.test")
    select_receipt = await actor.select(country["bid"], ["ca"])
    check_receipt = await actor.set_checked(terms["bid"], True)
    assert primary_receipt["success"] and primary_receipt["postconditions"]["value"] == "primary@example.test"
    assert secondary_receipt["success"] and secondary_receipt["postconditions"]["value"] == "secondary@example.test"
    assert select_receipt["success"] and select_receipt["postconditions"]["value"] == "ca"
    assert check_receipt["success"] and check_receipt["postconditions"]["checked"] is True

    trigger = one_element(observation, tag="button", text="Choose city")
    assert (await actor.click(trigger["bid"]))["success"]
    opened = await actor.observe()
    assert one_element(opened, role="listbox", label="City choices")
    paris = one_element(opened, role="option", text="Paris")
    assert (await actor.click(paris["bid"]))["success"]
    assert "City: Paris" in (await actor.extract("text"))["data"]

    replace = one_element(observation, tag="button", text="Replace field")
    assert (await actor.click(replace["bid"]))["success"]
    stale_receipt = await actor.fill(volatile["bid"], "must-not-write")
    assert stale_receipt["success"] is False
    assert stale_receipt["stale_bid"] is True
    assert "STALE_BID" in stale_receipt["error"]
    refreshed = await actor.observe()
    new_volatile = one_element(refreshed, tag="input", name="volatile")
    assert new_volatile["bid"] != volatile["bid"]
    assert (await actor.fill(new_volatile["bid"], "fresh-value"))["success"]

    delayed = one_element(refreshed, tag="button", text="Load delayed state")
    assert (await actor.click(delayed["bid"]))["success"]
    await actor.wait(700)
    delayed_observation = await actor.observe()
    assert one_element(delayed_observation, text="SPA ready")


async def test_nested_iframe_and_open_shadow_dom(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/frames")
    observation = await actor.observe()

    deep_input = one_element(observation, tag="input", name="deep-input")
    assert deep_input["frame"] >= 2
    assert deep_input["frame_url"].endswith("/frame-two")
    receipt = await actor.fill(deep_input["bid"], "nested-value")
    assert receipt["success"]
    assert receipt["postconditions"]["value"] == "nested-value"

    shadow_input = one_element(observation, tag="input", name="shadow-value")
    assert shadow_input["shadow"] is True
    assert (await actor.fill(shadow_input["bid"], "inside-shadow"))["success"]
    shadow_button = one_element(observation, tag="button", text="Save shadow")
    assert shadow_button["shadow"] is True
    assert (await actor.click(shadow_button["bid"]))["success"]
    updated = await actor.observe()
    shadow_result = one_element(updated, text="Shadow saved")
    assert shadow_result["shadow"] is True


async def test_pagination_and_virtual_list_exhaustion(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/collection")
    observed_items: set[str] = set()

    for page_number in range(1, 4):
        observation = await actor.observe()
        page_items = {
            element["text"]
            for element in observation["elements"]
            if element.get("tag") == "li" and element.get("text", "").startswith("Page item")
        }
        assert page_items == {f"Page item {(page_number - 1) * 2 + 1}", f"Page item {page_number * 2}"}
        observed_items.update(page_items)
        next_button = one_element(observation, tag="button", text="Next page")
        if page_number < 3:
            assert next_button["disabled"] is False
            assert (await actor.click(next_button["bid"]))["success"]
        else:
            assert next_button["disabled"] is True
    assert observed_items == {f"Page item {number}" for number in range(1, 7)}

    all_virtual: set[str] = set()
    bid_history: dict[str, set[str]] = {}
    previous_count = -1
    for _ in range(5):
        observation = await actor.observe()
        current = {
            element["text"]
            for element in observation["elements"]
            if element.get("role") == "listitem" and element.get("text", "").startswith("Virtual item")
        }
        all_virtual.update(current)
        for element in observation["elements"]:
            if element.get("role") == "listitem" and element.get("text", "").startswith("Virtual item"):
                bid_history.setdefault(element["bid"], set()).add(element["text"])
        virtual = one_element(observation, role="list", label="Virtual products")
        if len(all_virtual) == 15:
            before_terminal_scroll = len(all_virtual)
            await actor.scroll(4_000, virtual["bid"])
            await actor.wait(100)
            terminal = await actor.observe()
            terminal_items = {
                element["text"]
                for element in terminal["elements"]
                if element.get("role") == "listitem"
                and element.get("text", "").startswith("Virtual item")
            }
            assert len(all_virtual | terminal_items) == before_terminal_scroll
            assert one_element(terminal, text="End of virtual list")
            break
        assert len(all_virtual) > previous_count
        previous_count = len(all_virtual)
        scroll_receipt = await actor.scroll(160, virtual["bid"])
        assert scroll_receipt["success"]
        assert scroll_receipt["postconditions"]["after"] >= scroll_receipt["postconditions"]["before"]
        await actor.wait(100)
    assert all_virtual == {f"Virtual item {number}" for number in range(1, 16)}
    assert any(len(texts) > 1 for texts in bid_history.values()), "fixture must recycle DOM rows"


async def test_canvas_image_dialog_and_new_tab(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/visual-tabs")
    observation = await actor.observe()

    visual_results: list[tuple[Path, tuple[int, int]]] = []
    for target, question, expected_size in (
        (one_element(observation, tag="canvas", label="Quarterly chart"), "Which bars are shown?", (180, 100)),
        (one_element(observation, tag="img", label="Red test image"), "What color is the image?", (80, 60)),
    ):
        visual = await actor.visual_crop(target["bid"], question)
        path = Path(visual["path"])
        assert path.is_file() and path.stat().st_size > 0
        with Image.open(path) as image:
            assert all(abs(actual - expected) <= 1 for actual, expected in zip(image.size, expected_size))
            if target["tag"] == "img":
                red, green, blue = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
                assert red > 200 and green < 30 and blue < 30
        visual_results.append((path, expected_size))
    assert len({path for path, _ in visual_results}) == len(visual_results), "visual evidence must not be overwritten"

    alert = one_element(observation, tag="button", text="Show alert")
    await actor.arm_dialog("accept")
    alert_receipt = await actor.click(alert["bid"])
    assert alert_receipt["success"]
    assert alert_receipt["postconditions"]["dialog_events"] == [
        {"type": "alert", "message": "Proceed now", "action": "accept", "url": alert_receipt["before_url"]}
    ]

    prompt = one_element(observation, tag="button", text="Show prompt")
    await actor.arm_dialog("accept", "Codex")
    prompt_receipt = await actor.click(prompt["bid"])
    assert prompt_receipt["success"]
    assert prompt_receipt["postconditions"]["dialog_events"][0]["type"] == "prompt"
    assert "Saved Codex" in (await actor.extract("text"))["data"]

    new_tab_link = one_element(observation, tag="a", text="Open report tab")
    tab_receipt = await actor.click(new_tab_link["bid"])
    assert tab_receipt["success"]
    assert tab_receipt["postconditions"]["new_tab_count"] == 1
    assert tab_receipt["after_url"].endswith("/new-tab")
    tabs = await actor.tabs("list")
    assert len(tabs["tabs"]) == 2
    assert sum(tab["active"] for tab in tabs["tabs"]) == 1
    report_observation = await actor.observe()
    assert report_observation["title"] == "Report tab"
    original_index = next(tab["index"] for tab in tabs["tabs"] if tab["url"].endswith("/visual-tabs"))
    switched = await actor.tabs("switch", original_index)
    assert next(tab for tab in switched["tabs"] if tab["active"])["url"].endswith("/visual-tabs")


async def test_upload_download_pdf_document_and_browser_network(
    actor_factory: Any,
    tmp_path: Path,
) -> None:
    actor: BrowserActor = await actor_factory("/files-network")
    observation = await actor.observe()

    upload_source = tmp_path / "upload-proof.txt"
    upload_source.write_text("uploaded by BrowserActor", encoding="utf-8")
    upload = one_element(observation, tag="input", name="upload")
    upload_receipt = await actor.upload(upload["bid"], [str(upload_source)])
    assert upload_receipt["success"], upload_receipt["error"]
    assert upload_receipt["postconditions"]["files"] == ["upload-proof.txt"]
    upload_text = (await actor.extract("text"))["data"]
    assert "upload-proof.txt" in upload_text
    assert "uploaded by BrowserActor" in upload_text

    text_link = one_element(observation, tag="a", text="Download text")
    text_receipt = await actor.download(text_link["bid"])
    assert text_receipt["success"]
    text_path = Path(text_receipt["postconditions"]["download"]["path"])
    assert text_path.read_text(encoding="utf-8") == "downloaded through the browser\n"
    text_document = await actor.extract_document(str(text_path))
    assert text_document["pages"] == 1
    assert "downloaded through the browser" in text_document["text"]

    pdf_link = one_element(observation, tag="a", text="Download PDF")
    pdf_receipt = await actor.download(pdf_link["bid"])
    assert pdf_receipt["success"]
    pdf_path = Path(pdf_receipt["postconditions"]["download"]["path"])
    assert pdf_path.suffix == ".pdf"
    pdf_document = await actor.extract_document(str(pdf_path))
    assert pdf_document["pages"] == 1
    assert "Protocol III PDF evidence" in pdf_document["text"]

    scan_link = one_element(observation, tag="a", text="Download scanned PDF")
    scan_receipt = await actor.download(scan_link["bid"])
    assert scan_receipt["success"]
    scan_path = scan_receipt["postconditions"]["download"]["path"]
    scan_document = await actor.extract_document(scan_path)
    assert scan_document["pages"] == 1 and not scan_document["text"].strip()
    rendered = await actor.render_document_page(scan_path, 1, "Read the scanned page")
    rendered_path = Path(rendered["path"])
    assert rendered_path.is_file() and rendered_path.stat().st_size > 0
    with Image.open(rendered_path) as image:
        assert image.width > 100 and image.height > 100

    network_button = one_element(observation, tag="button", text="Load API data")
    network_receipt = await actor.click(network_button["bid"])
    assert network_receipt["success"]
    assert network_receipt["postconditions"]["network_response_count"] >= 1
    network = await actor.network_events(since_last=False)
    record = next(item for item in network["records"] if item["url"].endswith("/api/data"))
    assert record["resource_type"] == "fetch"
    assert record["method"] == "POST" and record["status"] == 200
    assert not any("authorization" in key.lower() or "api-key" in key.lower() for key in record["request_headers"])
    assert record["post_data"]["query"] == "browser-event"
    assert record["post_data"]["password"] == "[REDACTED]"
    assert record["body"]["message"] == "Network evidence ready"
    assert record["body"]["token"] == "[REDACTED]"
    await actor.flush_artifacts()
    capture = json.loads((actor.output_dir / "capture.json").read_text(encoding="utf-8"))
    assert any(item["url"].endswith("/api/data") for item in capture)
    serialized_capture = json.dumps(capture, ensure_ascii=False)
    for secret in ("Bearer hidden", "hidden-key", "hidden-password", "response-secret"):
        assert secret not in serialized_capture

    inline_link = one_element(observation, tag="a", text="Open inline PDF")
    inline_receipt = await actor.download(inline_link["bid"])
    assert inline_receipt["success"]
    inline_path = inline_receipt["postconditions"]["download"]["path"]
    inline_document = await actor.extract_document(inline_path)
    assert "Inline PDF evidence" in inline_document["text"]


async def test_browser_actor_serializes_playwright_on_owner_thread(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/thread")
    observation = await actor.observe()
    owner_thread = actor._owner_thread
    assert owner_thread is not None
    assert owner_thread != threading.get_ident()

    executor_threads = await asyncio.gather(
        *(actor._call(threading.get_ident) for _ in range(12))
    )
    assert set(executor_threads) == {owner_thread}
    with pytest.raises(RuntimeError):
        actor._assert_thread()

    field = one_element(observation, tag="input", name="thread-input")
    receipts = await asyncio.gather(
        actor.fill(field["bid"], "first"),
        actor.fill(field["bid"], "second"),
        actor.fill(field["bid"], "third"),
    )
    assert all(receipt["success"] for receipt in receipts)
    assert len({receipt["action_id"] for receipt in receipts}) == 3
    assert len({receipt["evidence_ids"][0] for receipt in receipts}) == 3
    trajectory = sorted((actor.output_dir / "trajectory").glob("*.png"))
    visual_trajectory = sorted((actor.output_dir / "trajectory_visual").glob("[0-9]*.png"))
    assert len(trajectory) >= 5  # start + observe + three action receipts
    assert [path.name for path in trajectory] == [path.name for path in visual_trajectory]


async def test_page_crash_fails_fast_without_success_receipt(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/thread")
    observation = await actor.observe()
    field = one_element(observation, tag="input", name="thread-input")

    await actor._call(actor._page._impl_obj.emit, "crash", None)
    started = time.monotonic()
    receipt = await actor.fill(field["bid"], "must-not-succeed")
    elapsed = time.monotonic() - started
    assert elapsed < 2
    assert receipt["success"] is False
    assert receipt["error"]


async def test_cdp_disconnect_fails_fast_without_success_receipt(actor_factory: Any) -> None:
    actor: BrowserActor = await actor_factory("/thread")
    observation = await actor.observe()
    field = one_element(observation, tag="input", name="thread-input")

    await actor._call(actor._browser.close)
    started = time.monotonic()
    receipt = await actor.fill(field["bid"], "must-not-succeed")
    elapsed = time.monotonic() - started
    assert elapsed < 2
    assert receipt["success"] is False
    assert receipt["error"]

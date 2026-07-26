from __future__ import annotations

import asyncio
import threading

import pytest

from web_agent.browser_actor import BrowserActor, _redact, _sanitize_headers
from web_agent.evidence import EvidenceStore


def test_sensitive_network_material_is_removed() -> None:
    assert _sanitize_headers(
        {
            "content-type": "application/json",
            "Authorization": "Bearer private",
            "Cookie": "session=private",
            "x-api-key": "private",
        }
    ) == {"content-type": "application/json"}
    assert _redact({"user": "alice", "password": "private", "nested": {"token": "private"}}) == {
        "user": "alice",
        "password": "[REDACTED]",
        "nested": {"token": "[REDACTED]"},
    }


def test_form_confirmation_requires_submit_intent_and_real_change() -> None:
    submit = {"target": {"tag": "button", "type": "submit", "text": "Submit"}}
    ordinary = {"target": {"tag": "button", "type": "button", "text": "Next"}}
    assert BrowserActor._is_confirmed_submission("click", True, True, submit, "Submitted successfully")
    assert not BrowserActor._is_confirmed_submission("fill", True, True, submit, "Submitted successfully")
    assert not BrowserActor._is_confirmed_submission("click", True, True, ordinary, "Success stories")
    assert not BrowserActor._is_confirmed_submission("click", True, False, submit, "Submitted successfully")


@pytest.mark.asyncio
async def test_eight_browser_actors_have_isolated_single_threads(tmp_path) -> None:
    actors = [
        BrowserActor(f"http://127.0.0.1:{9000 + index}", tmp_path / str(index), EvidenceStore())
        for index in range(8)
    ]

    def owner(actor: BrowserActor) -> int:
        actor._assert_thread()
        return threading.get_ident()

    thread_ids = await asyncio.gather(*(actor._call(owner, actor) for actor in actors))
    assert len(set(thread_ids)) == 8
    for index, actor in enumerate(actors):
        assert actor.output_dir == tmp_path / str(index)
        actor.output_dir.mkdir(parents=True)
    await asyncio.gather(*(actor.close() for actor in actors))


@pytest.mark.asyncio
async def test_cross_thread_access_fails_fast(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    await actor._call(actor._assert_thread)
    with pytest.raises(RuntimeError, match="线程边界"):
        actor._assert_thread()
    tmp_path.mkdir(exist_ok=True)
    await actor.close()

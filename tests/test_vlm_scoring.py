"""Label-likelihood scoring on a tiny random Gemma-3 with the real processor (needs Hub access)."""

import numpy as np
import pytest
from PIL import Image

from cxr import prompts
from cxr.tasks import get_task
from cxr.vlm import LabelScorer


@pytest.fixture(scope="module")
def tiny():
    try:
        from tiny_gemma3 import tiny_gemma3

        return tiny_gemma3()
    except Exception as exc:  # offline or incompatible transformers
        pytest.skip(f"tiny Gemma-3 unavailable: {exc}")


@pytest.mark.parametrize("task_name", ["task1", "task2"])
def test_prefix_cache_matches_full_forward(tiny, task_name):
    model, proc = tiny
    task = get_task(task_name)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (70, 90), dtype=np.uint8))
    naive = LabelScorer(model, proc, task.class_names, use_prefix_cache=False)
    cached = LabelScorer(model, proc, task.class_names, use_prefix_cache=True)
    for msgs, imgs in ((prompts.zero_shot_messages(task), [img]),
                       (prompts.icl_messages(task, list(task.class_names)), [img, img, img])):
        a, b = naive.score(msgs, imgs), cached.score(msgs, imgs)
        assert a.shape == (2,) and np.all(a < 0)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


def test_single_bos(tiny):
    model, proc = tiny
    task = get_task("task1")
    s = LabelScorer(model, proc, task.class_names)
    enc = s._encode([s._prompt_text(prompts.zero_shot_messages(task))],
                    [[Image.new("RGB", (32, 32))]])
    assert enc["input_ids"][0].tolist().count(proc.tokenizer.bos_token_id) == 1

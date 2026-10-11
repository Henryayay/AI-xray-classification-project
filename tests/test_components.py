import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxr import audit, prompts
from cxr.data import image_sha256, pad_to_square, stratified_subset
from cxr.icl import ExemplarSelector
from cxr.robustness import PERTURBATIONS, perturb
from cxr.tasks import TASKS, get_task
from cxr.vlm import parse_label


@pytest.mark.parametrize("name", sorted(TASKS))
def test_prompt_is_identical_for_training_and_inference(name):
    task = get_task(name)
    train = prompts.training_conversation(task, None, task.class_names[1])["messages"][0]
    infer = prompts.zero_shot_messages(task)[0]
    assert train["content"][0]["text"] == infer["content"][0]["text"]
    for c in task.class_names:
        assert f"'{c}'" in prompts.instruction(task)


def test_icl_messages_image_placeholders():
    task = get_task("task2")
    msgs = prompts.icl_messages(task, ["COVID-19", "Non-COVID Pneumonia", "COVID-19"])
    n_img = sum(c["type"] == "image" for c in msgs[0]["content"])
    assert n_img == 4 and msgs[0]["content"][-1]["type"] == "image"


@pytest.mark.parametrize("text,expected", [
    ("COVID-19", 1), ("Non-COVID Pneumonia", 0), ("non-covid pneumonia.", 0),
    ("Diagnosis: COVID-19", 1), ("unsure", -1), ("Pneumonia", -1),
])
def test_parse_label_task2(text, expected):
    assert parse_label(text, get_task("task2").class_names) == expected


def test_parse_label_task1():
    names = get_task("task1").class_names
    assert parse_label("Normal", names) == 0 and parse_label("Pneumonia\n", names) == 1


def test_stratified_subset():
    y = np.r_[np.zeros(100, int), np.ones(60, int)]
    idx = stratified_subset(y, 0.25, seed=3)
    assert (y[idx] == 0).sum() == 25 and (y[idx] == 1).sum() == 15
    assert np.array_equal(idx, stratified_subset(y, 0.25, seed=3))
    assert len(stratified_subset(y, 1.0, 0)) == 160


def test_pad_to_square_preserves_content():
    img = Image.new("L", (40, 20), 255)
    sq = pad_to_square(img)
    assert sq.size == (40, 40)
    a = np.asarray(sq)
    assert a[10:30].min() == 255 and a[:10].max() == 0


def test_image_hash_ignores_container():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (16, 16), dtype=np.uint8))
    assert image_sha256(img) == image_sha256(img.convert("RGB").convert("L"))


@pytest.mark.parametrize("kind", PERTURBATIONS)
def test_perturbations_keep_size_and_are_deterministic(kind):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 60), dtype=np.uint8))
    a, b = perturb(img, kind, seed=1), perturb(img, kind, seed=1)
    assert a.size == img.size and np.array_equal(np.asarray(a), np.asarray(b))


def test_centre_mask_removes_centre():
    img = Image.new("L", (100, 100), 200)
    a = np.asarray(perturb(img, "centre_mask"))
    assert a[50, 50] == 0 and a[2, 2] == 200


def test_random_selector_is_balanced_and_fixed():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    s = ExemplarSelector("random", y, 6, seed=1)
    a = s.select(query_key=0)
    assert np.array_equal(a, s.select(query_key=99))
    assert (y[a] == 0).sum() == 3
    assert not np.array_equal(a, ExemplarSelector("random", y, 6, seed=2).select())


def test_knn_selector_retrieves_nearest_per_class_and_excludes_duplicates():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 8))
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    q = emb[5] + 1e-3
    s = ExemplarSelector("knn", y, 4, seed=0, train_emb=emb)
    picks = s.select(q, query_key=1)
    assert 5 in picks and (y[picks] == 1).sum() == 2
    s2 = ExemplarSelector("knn", y, 4, seed=0, train_emb=emb, max_similarity=0.999)
    assert 5 not in s2.select(q, query_key=1)


def _ds_from_arrays(arrays, labels):
    class Split(list):
        pass

    recs = Split({"image": Image.fromarray(a), "label": int(lab)} for a, lab in zip(arrays, labels))
    return recs


def test_audit_finds_planted_duplicate_and_not_distinct_images():
    rng = np.random.default_rng(0)
    base = [rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(6)]
    smooth = []
    for b in base:
        a = np.asarray(Image.fromarray(b).resize((8, 8)).resize((64, 64), Image.BICUBIC), float)
        smooth.append(((a - a.min()) / np.ptp(a) * 255).astype(np.uint8))
    noisy = np.clip(smooth[0].astype(int) + rng.integers(-2, 3, (64, 64)), 0, 255).astype(np.uint8)
    ds = {"train": _ds_from_arrays(smooth[:4], [0, 1, 0, 1]),
          "validation": _ds_from_arrays(smooth[4:5], [0]),
          "test": _ds_from_arrays([noisy, smooth[5]], [0, 1])}
    rec, thumbs, d = audit.compute_records(ds)
    i, j, c = audit.candidate_pairs(d)
    pairs = audit.verified_pairs(rec, ds, i, j, coarse=c)
    assert len(pairs) == 1
    r = pairs.iloc[0]
    assert {r.split_a, r.split_b} == {"train", "test"} and r.grade == "near_identical"
    s = audit.summarise_duplicates(pairs)
    assert s["near_identical_or_exact"]["test_images_with_train_match"] == 1
    assert isinstance(audit.class_summary(rec), pd.DataFrame)

import pytest

torch = pytest.importorskip("torch")

from neuron_tweak.analysis import (
    ActivationCapture,
    ActivationStats,
    AnalysisError,
    capture_activations,
    global_top_neurons,
    plan_from_scores,
    score_neurons,
    top_neurons,
)


def make_stats(sums, sums_sq, counts, size, label):
    return ActivationStats(
        sums=[torch.tensor(vector, dtype=torch.float32) for vector in sums],
        sums_sq=[
            torch.tensor(vector, dtype=torch.float32) for vector in sums_sq
        ],
        counts=list(counts),
        intermediate_size=size,
        label=label,
    )


def test_means_and_variances():
    stats = make_stats([[2.0, 2.0]], [[4.0, 2.0]], [2], 2, "good")
    assert torch.allclose(stats.means(0), torch.tensor([1.0, 1.0]))
    assert torch.allclose(stats.variances(0), torch.tensor([1.0, 0.0]))


def test_score_neurons_ranks_contrastive_neurons():
    good = make_stats([[2.0, 2.0]], [[4.0, 2.0]], [2], 2, "good")
    bad = make_stats([[4.0, 2.0]], [[10.0, 2.0]], [2], 2, "bad")
    scores = score_neurons(good, bad)

    assert scores[0][0].cohen_d == pytest.approx(1.0, abs=1e-4)
    assert scores[0][1].cohen_d == pytest.approx(0.0, abs=1e-4)
    assert scores[0][0].mean_diff == pytest.approx(1.0)
    assert scores[0][0].address == "L1 N0"


def test_top_neurons_per_layer():
    good = make_stats(
        [[2.0, 2.0, 2.0]],
        [[4.0, 2.0, 2.0]],
        [2],
        3,
        "good",
    )
    bad = make_stats(
        [[6.0, 4.0, 2.0]],
        [[20.0, 10.0, 2.0]],
        [2],
        3,
        "bad",
    )
    scores = score_neurons(good, bad)

    top = top_neurons(scores, per_layer=2)
    assert [score.neuron for score in top] == [0, 1]
    assert global_top_neurons(scores, count=1)[0].neuron == 0


def test_plan_from_scores_uses_delta():
    good = make_stats([[0.0]], [[0.0]], [1], 1, "good")
    bad = make_stats([[2.0]], [[4.0]], [1], 1, "bad")
    scores = score_neurons(good, bad)

    plan = plan_from_scores(scores[0], delta=-0.5)
    edit = plan.get(1, 0)

    assert len(plan) == 1
    assert edit is not None
    assert edit.delta == -0.5
    assert edit.scale == 0.5


def test_activation_capture_records_hooks(runtime):
    capture = ActivationCapture(runtime, label="test")
    with capture:
        runtime.model(input_ids=torch.randint(0, 128, (2, 8)))

    stats = capture.stats
    assert stats.num_layers == 2
    assert stats.counts == [16, 16]
    assert stats.sums[0].shape == (32,)
    assert stats.label == "test"


class FakeTokenizer:
    chat_template = None
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, **kwargs):
        input_ids = torch.tensor([[1, 2, 3, 4] for _ in texts])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_capture_activations_batches_prompts(runtime):
    stats = capture_activations(
        runtime,
        FakeTokenizer(),
        ["one", "two", "three"],
        batch_size=2,
        label="bad",
    )
    assert stats.counts == [12, 12]


def test_capture_activations_requires_tokenizer(runtime):
    with pytest.raises(AnalysisError):
        capture_activations(runtime, None, ["prompt"])


def test_capture_rejects_empty_prompts(runtime):
    with pytest.raises(AnalysisError):
        capture_activations(runtime, FakeTokenizer(), [])


def test_score_rejects_mismatched_layers():
    good = make_stats([[0.0]], [[0.0]], [1], 1, "good")
    bad = make_stats([[0.0], [0.0]], [[0.0], [0.0]], [1, 1], 1, "bad")
    with pytest.raises(AnalysisError):
        score_neurons(good, bad)

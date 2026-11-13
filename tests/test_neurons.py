import pytest

from neuron_tweak.neurons import (
    NeuronEdit,
    NeuronPlan,
    NeuronSpecError,
    format_delta,
    parse_neuron_spec,
)


def test_parse_positive_steering():
    parsed = parse_neuron_spec("L14 N8949 +1")
    assert parsed.layer == 14
    assert parsed.neuron == 8949
    assert parsed.delta == 1.0


def test_parse_negative_steering():
    parsed = parse_neuron_spec("L13 N7956 -1")
    assert parsed.layer == 13
    assert parsed.neuron == 7956
    assert parsed.delta == -1.0


def test_parse_is_case_insensitive_and_tolerates_spacing():
    parsed = parse_neuron_spec("  l2n3+0.5  ")
    assert parsed.layer == 2
    assert parsed.neuron == 3
    assert parsed.delta == 0.5


def test_parse_delete_alias():
    parsed = parse_neuron_spec("L4 N10 delete")
    assert parsed.delta == -1.0


def test_parse_reset_alias():
    parsed = parse_neuron_spec("L4 N10 reset")
    assert parsed.delta is None


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "L14 N8949",
        "N8949 +1",
        "L0 N1 +1",
        "L1 N-1 +1",
        "L1 N1 +",
        "layer 1 neuron 2 +1",
    ],
)
def test_parse_rejects_invalid_specs(spec):
    with pytest.raises(NeuronSpecError):
        parse_neuron_spec(spec)


def test_scale_semantics():
    assert NeuronEdit(1, 0, 1.0).scale == 2.0
    assert NeuronEdit(1, 0, -1.0).scale == 0.0
    assert NeuronEdit(1, 0, -0.5).scale == 0.5
    assert NeuronEdit(1, 0, 0.0).scale == 1.0


def test_scale_is_clamped():
    assert NeuronEdit(1, 0, -5.0).scale == 0.0
    assert NeuronEdit(1, 0, 50.0, max_scale=10.0).scale == 10.0


def test_deletion_flag():
    assert NeuronEdit(1, 0, -1.0).is_deletion
    assert not NeuronEdit(1, 0, -0.5).is_deletion


def test_format_delta():
    assert format_delta(1.0) == "+1"
    assert format_delta(-1.0) == "-1"
    assert format_delta(0.5) == "+0.5"
    assert format_delta(-0.25) == "-0.25"


def test_plan_adds_and_overwrites_entries():
    plan = NeuronPlan()
    plan.add(NeuronEdit(14, 8949, 1.0))
    plan.add(NeuronEdit(13, 7956, -1.0))
    assert len(plan) == 2

    plan.add(NeuronEdit(14, 8949, -0.5))
    assert plan.get(14, 8949).delta == -0.5
    assert len(plan) == 2


def test_plan_from_text_supports_multiple_separators():
    plan = NeuronPlan.from_text("L14 N8949 +1, L13 N7956 -1\nL2 N3 +0.5; L4 N5 delete")
    assert len(plan) == 4
    assert plan.get(14, 8949).scale == 2.0
    assert plan.get(13, 7956).scale == 0.0
    assert plan.get(2, 3).scale == 1.5
    assert plan.get(4, 5).scale == 0.0


def test_plan_reset_removes_entry():
    plan = NeuronPlan.from_text("L14 N8949 +1")
    plan.add_specs("L14 N8949 reset")
    assert len(plan) == 0


def test_plan_remove():
    plan = NeuronPlan.from_text("L14 N8949 +1")
    assert plan.remove(14, 8949)
    assert not plan.remove(14, 8949)
    assert len(plan) == 0


def test_plan_round_trip():
    original = "L14 N8949 +1\nL13 N7956 -1\nL2 N3 +0.5"
    plan = NeuronPlan.from_text(original)
    assert plan.to_text() == "L2 N3 +0.5\nL13 N7956 -1\nL14 N8949 +1"


def test_effect_strings():
    assert NeuronEdit(1, 0, -1.0).effect == "delete"
    assert NeuronEdit(1, 0, -0.5).effect == "reduce x0.50"
    assert NeuronEdit(1, 0, 1.0).effect == "amplify x2.00"
    assert NeuronEdit(1, 0, 0.0).effect == "no change"

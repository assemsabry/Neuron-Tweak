from neuron_tweak.banner import banner_color, render_banner_text


def test_banner_has_five_rows():
    lines = render_banner_text()
    assert len(lines) == 5
    assert all(line.strip() for line in lines)


def test_banner_width_fits_terminal():
    lines = render_banner_text()
    assert max(len(line) for line in lines) <= 80


def test_banner_color_can_be_overridden(monkeypatch):
    monkeypatch.setenv("NEURON_TWEAK_BANNER_COLOR", "magenta")
    assert banner_color() == "magenta"

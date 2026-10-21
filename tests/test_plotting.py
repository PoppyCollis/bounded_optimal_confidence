"""The figures draw and save; the bar layout follows the model registry."""
import matplotlib
matplotlib.use("Agg")
import numpy as np

from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.plotting.bms_figures import pxp_bar_figure, table_figure
from bounded_optimal_confidence.plotting.model_bars import GROUP_GAP, bar_colors, bar_positions
from bounded_optimal_confidence.plotting.style import use_paper_style
from bounded_optimal_confidence.runs import load_fits
from bounded_optimal_confidence.stats.tables import bms_table


def test_bar_positions_leave_a_gap_at_each_group_change():
    x = bar_positions(list(MODELS))
    gaps = np.diff(x)
    assert np.allclose(gaps[[1, 5]], 1 + GROUP_GAP)             # Ent | hard, and hard | soft
    assert np.allclose(np.delete(gaps, [1, 5]), 1)


def test_bar_colors_follow_group_and_attention():
    c = bar_colors(list(MODELS))
    assert c[0] == c[1] and c[0][3] == 1.0                      # Diff and Ent: the opaque baseline colour
    assert c[2][:3] == c[6][:3] and c[2][3] == 0.75 and c[6][3] == 0.3     # hard and soft: same hue, different opacity


def test_bms_figures_are_written(tmp_path):
    use_paper_style()
    tab, bor = bms_table(load_fits(paper=True))
    table_figure(tab, bor, tmp_path / "table.png")
    pxp_bar_figure(tab, bor, tmp_path / "bars.png")
    assert (tmp_path / "table.png").stat().st_size > 10_000
    assert (tmp_path / "bars.png").stat().st_size > 10_000


def test_short_labels():
    assert [m.short_label for m in MODELS.values()][:4] == ["Diff", "Ent", "Sym Top-2", "Sym Spec"]
    assert MODELS["asym_spec_l_soft"].short_label == "Asym-L"

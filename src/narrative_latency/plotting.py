"""Shared matplotlib styling for CJK-safe charts."""
import matplotlib

# Sans-serif fonts that render Traditional Chinese without tofu boxes
# across macOS and Linux CI runners.
_CJK_FONTS = ["Arial Unicode MS", "PingFang TC", "Heiti TC", "sans-serif"]


def set_plot_style():
    """Apply CJK-safe sans-serif fonts and fix the unicode minus sign.

    Call once near the top of any script that renders Chinese text in charts,
    before creating figures.
    """
    matplotlib.rcParams["font.sans-serif"] = list(_CJK_FONTS)
    matplotlib.rcParams["axes.unicode_minus"] = False

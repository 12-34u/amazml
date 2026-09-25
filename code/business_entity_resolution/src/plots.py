"""Matplotlib helpers for the report notebooks.

Colours follow a fixed categorical order keyed by country, never by rank, so a country
keeps its colour in every chart. Countries not listed take the next free slot.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COUNTRY_ORDER = ["US", "India", "France"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def country_colors(countries) -> dict:
    order = COUNTRY_ORDER + sorted(set(countries) - set(COUNTRY_ORDER))
    return {c: SERIES[i % len(SERIES)] for i, c in enumerate(order)}


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, pct: bool = True, ax=None):
    """Bars for each index value, one series per column (countries)."""
    ax = ax or plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)[1]
    cols, colors = list(df.columns), country_colors(df.columns)
    width = 0.8 / len(cols)
    x = np.arange(len(df))
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * width, df[c].values, width * 0.9,
               color=colors[c], label=str(c), edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [str(v) for v in df.index])
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, labelcolor=INK2)
    return ax

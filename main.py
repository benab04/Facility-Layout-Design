"""
Integrated facility layout pipeline: ALDEP and CORELAP (construction) followed by
CRAFT (pairwise centroid interchange) on each layout, with reporting and plots.

Algorithm logic is taken from aldep.py, corelap.py, and craft.py without
altering the underlying procedures—placement and CRAFT loops match those
files; only orchestration, metrics, and plotting live here.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import corelap as cr
from aldep import ALDEP
from craft import craft_pairwise_exchange, rectilinear_distance, total_transport_cost

# ---------------------------------------------------------------------------
# Output directory for saved figures
# ---------------------------------------------------------------------------
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def _ensure_plot_dir() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)


def print_results_table(rows: List[Tuple[str, str, float, float, float]]) -> None:
    """Print the main numerical results in a terminal-friendly table."""
    headers = [
        "Seed",
        "Construction order",
        "Initial C",
        "CRAFT final C",
        "Visual replay C",
        "Improvement",
    ]
    table_rows = []
    for seed, order, initial_cost, final_cost, visual_cost in rows:
        improvement = initial_cost - final_cost
        table_rows.append(
            [
                seed,
                order,
                f"{initial_cost:.2f}",
                f"{final_cost:.2f}",
                f"{visual_cost:.2f}",
                f"{improvement:.2f}",
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    def fmt_row(values: List[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    print("=== Summary results table ===")
    print(fmt_row(headers))
    print(separator)
    for row in table_rows:
        print(fmt_row(row))


# --- Unified layout figure style (all construction + CRAFT grid plots) ---
LAYOUT_FIGSIZE = (7.0, 6.5)
CELL_EDGE_COLOR = "#1a1a1a"
CELL_EDGE_WIDTH = 1.2
GRID_LINE_COLOR = "#555555"
GRID_LINE_WIDTH = 0.45
CELL_LABEL_FONTSIZE = 7
# Fixed face color per department id (constant across all figures)
DEPT_FACE_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]


def department_face_color(dept_id: int) -> str:
    """Stable color for department index ``dept_id`` (same in ALDEP, CORELAP, CRAFT plots)."""
    if dept_id < 0 or dept_id >= len(DEPT_FACE_COLORS):
        return DEPT_FACE_COLORS[dept_id % len(DEPT_FACE_COLORS)]
    return DEPT_FACE_COLORS[dept_id]


# =============================================================================
# Shared realistic scenario (same departments, REL chart, and areas for both)
# =============================================================================
# Five-department machining cell: Machining, Assembly, Tooling, Testing, Shipping.
# REL chart matches corelap.py (CORELAP uses '-' on diagonal; ALDEP uses 'U').
SHARED_DEPARTMENTS_ORDER = [
    "Machining",
    "Assembly",
    "Tooling",
    "Testing",
    "Shipping",
]
SHARED_REL_CHART = [
    ["-", "A", "E", "I", "O"],
    ["A", "-", "U", "X", "I"],
    ["E", "U", "-", "A", "E"],
    ["I", "X", "A", "-", "U"],
    ["O", "I", "E", "U", "-"],
]
# Grid-cell counts per department (same as corelap.py)
SHARED_DEPT_SIZES = {0: 3, 1: 2, 2: 1, 3: 2, 4: 1}

# Realistic from-to flow (trips / period), rows = from, cols = to; indices 0..4
SHARED_FLOW = np.array(
    [
        [0, 12, 4, 9, 2],  # Machining -> others
        [3, 0, 1, 14, 11],  # Assembly
        [5, 2, 0, 3, 0],  # Tooling
        [4, 6, 0, 0, 8],  # Testing
        [2, 3, 0, 2, 0],  # Shipping
    ],
    dtype=float,
)


def rel_chart_to_aldep_matrix(rel_chart: List[List[str]]) -> List[List[str]]:
    """Diagonal '-' -> 'U' for ALDEP (letters only)."""
    n = len(rel_chart)
    out: List[List[str]] = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(rel_chart[i][j] if i != j else "U")
        out.append(row)
    return out


def areas_int_keys_to_aldep_names(
    dept_sizes: Dict[int, int], names: List[str]
) -> Dict[str, int]:
    return {names[i]: dept_sizes[i] for i in range(len(names))}


# =============================================================================
# Geometry: centroids for CRAFT / transport (cell centers, x = col, y = row)
# =============================================================================
def centroids_from_corelap_grid(grid: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """
    Centroid of each department's cells in **layout coordinates**:
    (x, y) = (column + 0.5, row + 0.5) with row 0 at the top and y increasing downward.
    Same convention as `craft.py` / rectilinear_distance; used only for cost, not for Matplotlib directly.
    """
    empty = -1
    buckets: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(5)}
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            if v == empty:
                continue
            d = int(v)
            buckets[d].append((c + 0.5, r + 0.5))
    return {
        d: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for d, pts in buckets.items()
        if pts
    }


def centroids_from_aldep_grid(
    grid: np.ndarray, names_in_order: List[str]
) -> Dict[int, Tuple[float, float]]:
    name_to_id = {name: i for i, name in enumerate(names_in_order)}
    buckets: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(names_in_order))}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            cell = grid[r, c]
            if cell is None:
                continue
            d = name_to_id[str(cell)]
            buckets[d].append((c + 0.5, r + 0.5))
    return {
        d: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for d, pts in buckets.items()
        if pts
    }


def layout_xy_to_display_xy(cx: float, cy_layout: float, grid_height: int) -> Tuple[float, float]:
    """Map layout (x right, y down from top) to Matplotlib coords used by cell patches."""
    return (cx, float(grid_height) - cy_layout)


def cell_counts_from_grid(grid: np.ndarray, empty: int = -1) -> Dict[int, int]:
    """Count occupied cells per department id on the grid."""
    counts: Dict[int, int] = {}
    for v in grid.flat:
        if v == empty:
            continue
        d = int(v)
        counts[d] = counts.get(d, 0) + 1
    return counts


def repartition_swap_pair_visual(grid: np.ndarray, a: int, b: int, empty: int = -1) -> np.ndarray:
    """
    Physical visualization of one CRAFT pairwise move on the current cell grid.

    Equal areas: swap all labels ``a`` and ``b``.

    Unequal areas (k = smaller cell count): assign ``k`` cells of the **larger**
    department that lie farthest (Manhattan) from the **smaller** department’s
    cells to the smaller id; the rest of the union of ``a`` and ``b`` becomes
    the larger id (k cells carved from the interior of the large region, away
    from the common boundary).
    """
    g = grid.copy()
    pos_a = list(zip(*np.where(g == a)))
    pos_b = list(zip(*np.where(g == b)))
    if not pos_a or not pos_b:
        return g
    na, nb = len(pos_a), len(pos_b)
    if na == nb:
        return np.where(g == a, b, np.where(g == b, a, g))
    if na > nb:
        large_id, small_id, pos_large, pos_small = a, b, pos_a, pos_b
    else:
        large_id, small_id, pos_large, pos_small = b, a, pos_b, pos_a
    k = len(pos_small)

    def dist_to_small(rc: Tuple[int, int]) -> int:
        r0, c0 = rc
        return min(abs(r0 - r) + abs(c0 - c) for (r, c) in pos_small)

    pos_large_sorted = sorted(pos_large, key=lambda rc: (-dist_to_small(rc), rc[0], rc[1]))
    chosen_for_small = set(pos_large_sorted[:k])
    union = set(pos_large) | set(pos_small)
    for r, c in union:
        g[r, c] = small_id if (r, c) in chosen_for_small else large_id
    return g


def apply_visual_craft_swaps(grid: np.ndarray, craft_history: List[dict], empty: int = -1) -> np.ndarray:
    """Replay CRAFT ``selected_pair`` sequence using ``repartition_swap_pair_visual``."""
    g = grid.copy()
    for rec in craft_history:
        pair = rec.get("selected_pair")
        if pair is None:
            continue
        g = repartition_swap_pair_visual(g, int(pair[0]), int(pair[1]), empty=empty)
    return g


def geometric_centroids_layout_xy(
    grid: np.ndarray, empty: int = -1
) -> Dict[int, Tuple[float, float]]:
    """Same (x, y) convention as centroids_from_corelap_grid / CRAFT."""
    buckets: Dict[int, List[Tuple[float, float]]] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v == empty:
                continue
            d = int(v)
            buckets.setdefault(d, []).append((c + 0.5, r + 0.5))
    return {
        d: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for d, pts in buckets.items()
        if pts
    }


def aldep_grid_to_int_grid(grid: np.ndarray, names_in_order: List[str]) -> np.ndarray:
    """ALDEP cell values (dept names) -> integer ids 0..n-1, empty -> -1."""
    name_to_id = {name: i for i, name in enumerate(names_in_order)}
    h, w = grid.shape
    out = np.full((h, w), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            cell = grid[r, c]
            if cell is None:
                continue
            out[r, c] = name_to_id[str(cell)]
    return out


def partial_transport_cost(
    flow: np.ndarray, centroids: Dict[int, Tuple[float, float]]
) -> float:
    """Transport cost among departments that have a centroid (placed on grid)."""
    ids = sorted(centroids.keys())
    total = 0.0
    for i in ids:
        for j in ids:
            if i == j:
                continue
            total += flow[i, j] * rectilinear_distance(centroids[i], centroids[j])
    return float(total)


def all_pair_feasible(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# =============================================================================
# ALDEP: same logic as aldep.ALDEP.place_departments / generate_sequence (no edits)
# =============================================================================
def run_aldep_tracked(
    departments: List[str],
    areas: Dict[str, int],
    rel_matrix: List[List[str]],
    facility_width: int,
    mdp_letter: str,
    flow: np.ndarray,
) -> Tuple[ALDEP, List[str], List[float], List[float]]:
    """
    Runs ALDEP.generate_sequence and placement sweep identical to ALDEP.place_departments,
    recording after each department in the sequence is fully placed:
    - ALDEP adjacency layout score (calculate_layout_score)
    - Partial material-handling cost among departments with cells placed so far
    """
    aldep = ALDEP(departments, areas, rel_matrix, facility_width)
    sequence = aldep.generate_sequence(mdp_letter=mdp_letter)

    rel_scores: List[float] = []
    transport_partials: List[float] = []

    x, y = 0, 0
    direction = 1
    for dept in sequence:
        blocks_needed = aldep.areas[dept]
        while blocks_needed > 0:
            aldep.grid[y][x] = dept
            blocks_needed -= 1
            y += direction
            if y >= aldep.facility_height or y < 0:
                direction *= -1
                y += direction
                x += 1
                if x >= aldep.facility_width:
                    break
        rel_scores.append(aldep.calculate_layout_score())
        c = centroids_from_aldep_grid(aldep.grid, departments)
        transport_partials.append(partial_transport_cost(flow, c))

    return aldep, sequence, rel_scores, transport_partials


# =============================================================================
# CORELAP: configure module globals then run placement loop from corelap.py
# =============================================================================
def configure_corelap_shared_scenario() -> None:
    cr.departments = [0, 1, 2, 3, 4]
    cr.dept_names = list(SHARED_DEPARTMENTS_ORDER)
    cr.dept_sizes = dict(SHARED_DEPT_SIZES)
    cr.rel_chart = [list(row) for row in SHARED_REL_CHART]


def run_corelap_tracked(
    grid_size: int, flow: np.ndarray
) -> Tuple[np.ndarray, List[int], List[float]]:
    """
    Same two-phase procedure as corelap.py __main__: TCR / placement order, then
    multi-grid placement loop (center start, boundary cells, PR maximization).
    Records partial transport cost after each full department is placed.
    """
    configure_corelap_shared_scenario()
    tcrs = cr.calculate_tcr()
    placement_order = cr.get_placement_order(tcrs)
    layout_grid = np.full((grid_size, grid_size), -1)
    transport_partials: List[float] = []

    for step, dept in enumerate(placement_order):
        size_needed = cr.dept_sizes[dept]
        for block_num in range(size_needed):
            if step == 0 and block_num == 0:
                center = grid_size // 2
                layout_grid[center, center] = dept
                continue
            if block_num == 0:
                candidate_locations = cr.get_boundary_cells(layout_grid)
            else:
                candidate_locations = cr.get_boundary_cells(layout_grid, restrict_to_dept=dept)

            best_loc = None
            max_pr = -float("inf")
            for r, c in candidate_locations:
                pr_score = 0
                neighbors = cr.get_neighbors(layout_grid, r, c)
                for nr, nc in neighbors:
                    adj_dept = layout_grid[nr, nc]
                    if adj_dept != -1 and adj_dept != dept:
                        rel = cr.rel_chart[dept][adj_dept]
                        pr_score += cr.pr_weights[rel]
                if pr_score > max_pr:
                    max_pr = pr_score
                    best_loc = (r, c)
            layout_grid[best_loc[0], best_loc[1]] = dept

        c = centroids_from_corelap_grid(layout_grid)
        transport_partials.append(partial_transport_cost(flow, c))

    return layout_grid, placement_order, transport_partials


# =============================================================================
# Plotting (saved PNGs) — unified grid style for ALDEP, CORELAP, and CRAFT
# =============================================================================
def _draw_unified_layout_on_ax(
    ax,
    grid: np.ndarray,
    names: List[str],
    *,
    empty: int = -1,
    centroids: Optional[Dict[int, Tuple[float, float]]] = None,
    title: str = "",
    show_axis_labels: bool = True,
) -> None:
    """Draw one facility grid: colored cells + labels; optional centroid markers."""
    h, w = grid.shape
    counts = cell_counts_from_grid(grid, empty=empty)
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v == empty:
                continue
            di = int(v)
            disp_x = float(c)
            disp_y = float(h - 1 - r)
            face = department_face_color(di)
            ax.add_patch(
                plt.Rectangle(
                    (disp_x, disp_y),
                    1.0,
                    1.0,
                    facecolor=face,
                    edgecolor=CELL_EDGE_COLOR,
                    linewidth=CELL_EDGE_WIDTH,
                )
            )
            nc = counts.get(di, 0)
            label = f"{names[di]}\n({nc} cells)"
            ax.text(
                disp_x + 0.5,
                disp_y + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=CELL_LABEL_FONTSIZE,
                color="white",
                fontweight="bold",
                linespacing=0.95,
            )
    if centroids:
        # Geometric centroids of current cell ownership (layout coords)
        geo_layout = geometric_centroids_layout_xy(grid, empty=empty)
        for di in sorted(centroids):
            cx_l, cy_l = centroids[di]
            x_d, y_d = layout_xy_to_display_xy(cx_l, cy_l, h)
            ax.plot(
                x_d,
                y_d,
                "o",
                ms=7,
                mfc="white",
                mec=department_face_color(di),
                mew=1.2,
                zorder=6,
            )
            # Label = department that **owns** this CRAFT representative point
            t = ax.annotate(
                names[di],
                (x_d, y_d),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=department_face_color(di),
                zorder=7,
            )
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
            if di in geo_layout:
                gx_l, gy_l = geo_layout[di]
                gx_d, gy_d = layout_xy_to_display_xy(gx_l, gy_l, h)
                if (abs(gx_d - x_d) + abs(gy_d - y_d)) > 0.08:
                    ax.plot(
                        [gx_d, x_d],
                        [gy_d, y_d],
                        color=CELL_EDGE_COLOR,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.75,
                        zorder=5,
                    )
                    ax.plot(gx_d, gy_d, "+", ms=8, mew=1.2, color=CELL_EDGE_COLOR, zorder=5)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks(range(w + 1))
    ax.set_yticks(range(h + 1))
    ax.grid(True, color=GRID_LINE_COLOR, linewidth=GRID_LINE_WIDTH, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    if show_axis_labels:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0=top)")


def plot_unified_layout_grid(
    grid: np.ndarray,
    names: List[str],
    title: str,
    filename: str,
    *,
    empty: int = -1,
    centroids: Optional[Dict[int, Tuple[float, float]]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=LAYOUT_FIGSIZE)
    _draw_unified_layout_on_ax(
        ax, grid, names, empty=empty, centroids=centroids, title=title
    )
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)


def plot_cost_series(
    xs,
    ys,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    y2=None,
    y2label=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "o-", label=ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y2 is not None and y2label:
        ax2 = ax.twinx()
        ax2.plot(xs, y2, "s--", color="tab:orange", label=y2label)
        ax2.set_ylabel(y2label)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)


def plot_bar_comparison(labels: List[str], values: List[float], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["tab:blue", "tab:green"])
    ax.set_ylabel("Total transport cost")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)


def plot_four_layouts(
    grid_aldep_int: np.ndarray,
    grid_corelap: np.ndarray,
    grid_craft_aldep_vis: np.ndarray,
    grid_craft_core_vis: np.ndarray,
    names: List[str],
    aldep_score: float,
    cost_core_init: float,
    geo_centroids_aldep: Dict[int, Tuple[float, float]],
    geo_centroids_core: Dict[int, Tuple[float, float]],
    cost_a: float,
    cost_c: float,
    filename: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    _draw_unified_layout_on_ax(
        axes[0, 0],
        grid_aldep_int,
        names,
        title=f"ALDEP initial (REL score={aldep_score:.0f})",
        show_axis_labels=False,
    )
    _draw_unified_layout_on_ax(
        axes[0, 1],
        grid_corelap,
        names,
        title=f"CORELAP initial (transport {cost_core_init:.1f})",
        show_axis_labels=False,
    )
    _draw_unified_layout_on_ax(
        axes[1, 0],
        grid_craft_aldep_vis,
        names,
        centroids=geo_centroids_aldep,
        title=f"CRAFT visual (ALDEP seed), cost={cost_a:.1f}",
        show_axis_labels=False,
    )
    _draw_unified_layout_on_ax(
        axes[1, 1],
        grid_craft_core_vis,
        names,
        centroids=geo_centroids_core,
        title=f"CRAFT visual (CORELAP seed), cost={cost_c:.1f}",
        show_axis_labels=False,
    )
    fig.suptitle(
        "Construction vs CRAFT (visual repartition; rings = geometric centroids)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)


# =============================================================================
# Main driver
# =============================================================================
def main() -> None:
    _ensure_plot_dir()
    random.seed(42)
    np.random.seed(42)

    names = SHARED_DEPARTMENTS_ORDER
    flow = SHARED_FLOW
    rel_aldep = rel_chart_to_aldep_matrix(SHARED_REL_CHART)
    areas_aldep = areas_int_keys_to_aldep_names(SHARED_DEPT_SIZES, names)
    facility_width = 4

    print("=== Shared scenario ===")
    print("Departments:", names)
    print("Areas (grid cells):", SHARED_DEPT_SIZES)
    print("Flow matrix (from-to):\n", flow)
    print()

    # --- ALDEP ---
    print("--- ALDEP (class + sequence + sweep from aldep.py) ---")
    aldep, seq_aldep, rel_curve, transp_aldep_curve = run_aldep_tracked(
        names, areas_aldep, rel_aldep, facility_width, "I", flow
    )
    score_aldep = aldep.calculate_layout_score()
    print("Placement sequence:", seq_aldep)
    print("Final ALDEP REL adjacency score:", score_aldep)
    cents_aldep_init = centroids_from_aldep_grid(aldep.grid, names)
    cost_aldep_init = total_transport_cost(flow, cents_aldep_init)
    print("Transport cost at ALDEP centroids (before CRAFT):", f"{cost_aldep_init:.2f}")
    print()

    # --- CORELAP ---
    print("--- CORELAP (TCR / order + placement loop from corelap.py) ---")
    configure_corelap_shared_scenario()
    tcrs = cr.calculate_tcr()
    print("TCR values:", {cr.dept_names[k]: v for k, v in tcrs.items()})
    grid_corelap, placement_order, transp_core_curve = run_corelap_tracked(7, flow)
    print("Placement order:", [cr.dept_names[d] for d in placement_order])
    cents_core_init = centroids_from_corelap_grid(grid_corelap)
    cost_core_init = total_transport_cost(flow, cents_core_init)
    print("Transport cost at CORELAP centroids (before CRAFT):", f"{cost_core_init:.2f}")
    print()

    feasible = all_pair_feasible(5)

    # --- CRAFT from each construction solution ---
    print("--- CRAFT (craft_pairwise_exchange from craft.py) ---")
    hist_craft_aldep = craft_pairwise_exchange(
        flow=flow,
        initial_centroids=cents_aldep_init,
        feasible_pairs=feasible,
        max_iterations=50,
    )
    hist_craft_core = craft_pairwise_exchange(
        flow=flow,
        initial_centroids=cents_core_init,
        feasible_pairs=feasible,
        max_iterations=50,
    )

    cost_final_aldep = hist_craft_aldep[-1]["cost"]
    cost_final_core = hist_craft_core[-1]["cost"]

    print("Final transport cost (CRAFT seeded from ALDEP):", f"{cost_final_aldep:.2f}")
    print("Final transport cost (CRAFT seeded from CORELAP):", f"{cost_final_core:.2f}")
    print()

    grid_aldep_int = aldep_grid_to_int_grid(aldep.grid, names)
    # Physical visualization of CRAFT swap sequence (does not change craft.py logic)
    grid_craft_aldep_vis = apply_visual_craft_swaps(grid_aldep_int.copy(), hist_craft_aldep)
    grid_craft_core_vis = apply_visual_craft_swaps(grid_corelap.copy(), hist_craft_core)
    geo_craft_aldep = centroids_from_corelap_grid(grid_craft_aldep_vis)
    geo_craft_core = centroids_from_corelap_grid(grid_craft_core_vis)
    cost_vis_aldep = total_transport_cost(flow, geo_craft_aldep)
    cost_vis_core = total_transport_cost(flow, geo_craft_core)
    print(
        "Transport cost from visual-replay layout centroids (ALDEP seed):",
        f"{cost_vis_aldep:.2f}",
    )
    print(
        "Transport cost from visual-replay layout centroids (CORELAP seed):",
        f"{cost_vis_core:.2f}",
    )
    print("(Abstract CRAFT costs above may differ; plots use the repartition rule.)")
    print()

    print_results_table(
        [
            (
                "ALDEP",
                " -> ".join(seq_aldep),
                cost_aldep_init,
                cost_final_aldep,
                cost_vis_aldep,
            ),
            (
                "CORELAP",
                " -> ".join(cr.dept_names[d] for d in placement_order),
                cost_core_init,
                cost_final_core,
                cost_vis_core,
            ),
        ]
    )
    print()

    # --- Save plots (unified rectangle + grid style) ---
    plot_unified_layout_grid(
        grid_aldep_int,
        names,
        f"ALDEP initial layout (REL score={score_aldep:.0f})",
        "01_aldep_initial_layout.png",
    )
    plot_unified_layout_grid(
        grid_corelap,
        names,
        f"CORELAP initial layout (transport {cost_core_init:.1f})",
        "02_corelap_initial_layout.png",
    )
    plot_unified_layout_grid(
        grid_craft_aldep_vis,
        names,
        f"CRAFT visual (ALDEP seed): repartition rule; abstract cost={cost_final_aldep:.1f}",
        "03_craft_final_from_aldep.png",
        centroids=geo_craft_aldep,
    )
    plot_unified_layout_grid(
        grid_craft_core_vis,
        names,
        f"CRAFT visual (CORELAP seed): repartition rule; abstract cost={cost_final_core:.1f}",
        "04_craft_final_from_corelap.png",
        centroids=geo_craft_core,
    )

    steps_aldep = list(range(1, len(rel_curve) + 1))
    plot_cost_series(
        steps_aldep,
        rel_curve,
        "Departments placed (order in ALDEP sequence)",
        "ALDEP REL adjacency score",
        "ALDEP: REL score vs construction progress",
        "05_aldep_rel_score_vs_steps.png",
        y2=transp_aldep_curve,
        y2label="Partial transport cost",
    )
    steps_core = list(range(1, len(transp_core_curve) + 1))
    plot_cost_series(
        steps_core,
        transp_core_curve,
        "Departments placed (CORELAP order)",
        "Partial transport cost",
        "CORELAP: partial transport cost vs placement steps",
        "06_corelap_transport_vs_steps.png",
    )

    it_a = [h["iteration"] for h in hist_craft_aldep if "cost" in h]
    c_a = [h["cost"] for h in hist_craft_aldep if "cost" in h]
    plot_cost_series(
        it_a,
        c_a,
        "CRAFT iteration",
        "Total transport cost",
        "CRAFT cost curve (initial layout from ALDEP)",
        "07_craft_cost_vs_iteration_aldep_seed.png",
    )
    it_c = [h["iteration"] for h in hist_craft_core if "cost" in h]
    c_c = [h["cost"] for h in hist_craft_core if "cost" in h]
    plot_cost_series(
        it_c,
        c_c,
        "CRAFT iteration",
        "Total transport cost",
        "CRAFT cost curve (initial layout from CORELAP)",
        "08_craft_cost_vs_iteration_corelap_seed.png",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(it_a, c_a, "o-", label=f"CRAFT from ALDEP (final {cost_final_aldep:.1f})")
    ax.plot(it_c, c_c, "s-", label=f"CRAFT from CORELAP (final {cost_final_core:.1f})")
    ax.set_xlabel("CRAFT iteration")
    ax.set_ylabel("Total transport cost")
    ax.set_title("CRAFT: cost vs iteration (both seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "09_craft_cost_both_seeds.png"), dpi=150)
    plt.close(fig)

    plot_bar_comparison(
        ["CRAFT\n(ALDEP seed)", "CRAFT\n(CORELAP seed)"],
        [cost_final_aldep, cost_final_core],
        "Final transport cost after CRAFT",
        "10_final_cost_comparison.png",
    )

    plot_four_layouts(
        grid_aldep_int,
        grid_corelap,
        grid_craft_aldep_vis,
        grid_craft_core_vis,
        names,
        score_aldep,
        cost_core_init,
        geo_craft_aldep,
        geo_craft_core,
        cost_final_aldep,
        cost_final_core,
        "11_overview_four_panels.png",
    )

    print(f"All figures saved under: {PLOT_DIR}")


if __name__ == "__main__":
    main()

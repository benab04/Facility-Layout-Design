from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ============================================================
# CRAFT (centroid-based teaching implementation)
# ------------------------------------------------------------
# This version follows the computational logic shown in the PDF:
#     1) compute transportation cost from a from-to flow matrix
#     2) evaluate all feasible pairwise swaps
#     3) choose the swap that gives the biggest cost reduction
#
# It is written to match the worked example in the slides.
# ============================================================


@dataclass(frozen=True)
class Department:
    dept_id: int
    area: float
    centroid: Tuple[float, float]  # (x, y)


def rectilinear_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Manhattan / rectilinear distance."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def total_transport_cost(
    flow: np.ndarray,
    centroids: Dict[int, Tuple[float, float]],
    cost_per_unit_distance: float = 1.0,
) -> float:
    """
    Total cost = sum over ordered pairs i != j of:
    flow[i, j] * cost_per_unit_distance * rectilinear_distance(centroid_i, centroid_j)
    In the PDF example, c_ij = 1 for all i != j.
    """
    dept_ids = sorted(centroids.keys())
    index = {dept_id: pos for pos, dept_id in enumerate(dept_ids)}
    total = 0.0
    for i in dept_ids:
        for j in dept_ids:
            if i == j:
                continue
            fij = flow[index[i], index[j]]
            dij = rectilinear_distance(centroids[i], centroids[j])
            total += fij * cost_per_unit_distance * dij
    return float(total)


def swap_centroids(
    centroids: Dict[int, Tuple[float, float]],
    a: int,
    b: int,
) -> Dict[int, Tuple[float, float]]:
    """Return a new centroid map after swapping departments a and b."""
    new_centroids = dict(centroids)
    new_centroids[a], new_centroids[b] = new_centroids[b], new_centroids[a]
    return new_centroids


def craft_pairwise_exchange(
    flow: np.ndarray,
    initial_centroids: Dict[int, Tuple[float, float]],
    feasible_pairs: Iterable[Tuple[int, int]],
    cost_per_unit_distance: float = 1.0,
    max_iterations: int = 50,
) -> List[dict]:
    """
    Greedy CRAFT-style improvement loop.
    At each iteration:
    - evaluate all feasible pair swaps
    - choose the swap with the lowest transportation cost
    - accept it only if it improves the current layout
    Returns a history list with:
    iteration, layout, cost, selected_pair, candidate_table
    """
    # Normalize pairs so (a, b) and (b, a) are the same.
    candidate_pairs = sorted({tuple(sorted(p)) for p in feasible_pairs})
    current = dict(initial_centroids)
    current_cost = total_transport_cost(flow, current, cost_per_unit_distance)
    history: List[dict] = [
        {
            "iteration": 0,
            "layout": dict(current),
            "cost": current_cost,
            "selected_pair": None,
            "candidate_table": [],
        }
    ]

    seen_layouts = {tuple(sorted(current.items()))}
    for it in range(1, max_iterations + 1):
        candidate_table: List[Tuple[int, int, float, float]] = []
        best_pair: Optional[Tuple[int, int]] = None
        best_layout: Optional[Dict[int, Tuple[float, float]]] = None
        best_cost = current_cost
        for a, b in candidate_pairs:
            trial = swap_centroids(current, a, b)
            trial_cost = total_transport_cost(flow, trial, cost_per_unit_distance)
            gain = current_cost - trial_cost
            candidate_table.append((a, b, trial_cost, gain))
            if trial_cost < best_cost - 1e-12:
                best_cost = trial_cost
                best_pair = (a, b)
                best_layout = trial
        candidate_table.sort(key=lambda row: row[2])
        # Stop if no improving swap exists.
        if best_pair is None:
            history.append(
                {
                    "iteration": it,
                    "layout": dict(current),
                    "cost": current_cost,
                    "selected_pair": None,
                    "candidate_table": candidate_table,
                    "stopped": True,
                    "reason": "no_improving_swap",
                }
            )
            break
        layout_key = tuple(sorted(best_layout.items()))
        if layout_key in seen_layouts:
            history.append(
                {
                    "iteration": it,
                    "layout": dict(current),
                    "cost": current_cost,
                    "selected_pair": None,
                    "candidate_table": candidate_table,
                    "stopped": True,
                    "reason": "cycle_detected",
                }
            )
            break
        current = best_layout
        current_cost = best_cost
        seen_layouts.add(layout_key)
        history.append(
            {
                "iteration": it,
                "layout": dict(current),
                "cost": current_cost,
                "selected_pair": best_pair,
                "candidate_table": candidate_table,
            }
        )
    return history


def pretty_print_candidate_table(candidate_table: List[Tuple[int, int, float, float]]) -> None:
    print("pair      swapped-cost    gain")
    print("----      ------------    ----")
    for a, b, cost, gain in candidate_table:
        print(f"{a}-{b:<2}      {cost:12.1f}    {gain:6.1f}")


def pretty_print_layout(layout: Dict[int, Tuple[float, float]]) -> None:
    for dept_id in sorted(layout):
        x, y = layout[dept_id]
        print(f"Dept {dept_id}: centroid = ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    # --------------------------------------------------------
    # Dummy dataset = the exact 5-department example from the PDF
    # --------------------------------------------------------
    # Flow matrix (from-to chart) from the slide:
    # 1 -> 2: 5, 1 -> 3: 2, 1 -> 4: 4
    # 2 -> 3: 2, 2 -> 4: 5
    # 3 -> 1: 2, 3 -> 5: 5
    # 4 -> 1: 3, 4 -> 3: 1
    # 5 -> 3: 2
    #
    # The PDF uses cij = 1 for all i != j.
    flow = np.array(
        [
            [0, 5, 2, 4, 0],
            [0, 0, 2, 5, 0],
            [2, 0, 0, 0, 5],
            [3, 0, 1, 0, 0],
            [0, 0, 2, 0, 0],
        ],
        dtype=float,
    )

    # Initial centroids from the PDF
    initial_centroids = {
        1: (2.0, 6.0),
        2: (2.0, 2.0),
        3: (7.0, 2.0),
        4: (8.0, 6.0),
        5: (5.0, 6.0),
    }
    # Feasible interchanges shown in the PDF:
    # 1-2, 1-4, 1-5, 2-3, 2-4, 3-4, 3-5, 4-5
    feasible_pairs = [
        (1, 2),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
    ]
    # One improvement pass is enough to reproduce the worked-table result
    # where the best swap is 4-5.
    history = craft_pairwise_exchange(
        flow=flow,
        initial_centroids=initial_centroids,
        feasible_pairs=feasible_pairs,
        max_iterations=1,
    )
    print("\nINITIAL LAYOUT")
    pretty_print_layout(history[0]["layout"])
    print(f"Initial cost = {history[0]['cost']:.1f}")
    print("\nCANDIDATE SWAPS (iteration 1)")
    pretty_print_candidate_table(history[1]["candidate_table"])
    print("\nBEST MOVE ACCEPTED")
    print(f"Selected pair = {history[1]['selected_pair']}")
    print(f"New cost = {history[1]['cost']:.1f}")
    print("\nNEW LAYOUT")
    pretty_print_layout(history[1]["layout"])

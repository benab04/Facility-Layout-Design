# Facility layout project: ALDEP, CORELAP, and CRAFT

This repository contains standalone implementations of **ALDEP** (`aldep.py`), **CORELAP** (`corelap.py`), and **CRAFT** (`craft.py`), plus an integrated driver **`main.py`** that applies all three to one shared layout scenario and compares outcomes.

## Prerequisites

Python 3.9+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## What `main.py` does

1. **Shared scenario**  
   Five departments (Machining, Assembly, Tooling, Testing, Shipping) with the same REL chart as in `corelap.py`, the same grid-cell areas `{0:3, 1:2, 2:1, 3:2, 4:1}`, and a realistic **from-to flow matrix** for material handling. The REL chart is converted to an ALDEP-style letter matrix by replacing diagonal `'-'` with `'U'`.

2. **ALDEP** (`aldep.py`)  
   Instantiates `ALDEP`, runs `generate_sequence(mdp_letter='I')`, and places departments using the same sweep logic as `ALDEP.place_departments` (column sweep with direction reversal). After each department in the sequence is fully placed, the driver records:
   - **REL adjacency score** from `calculate_layout_score()` (ALDEP’s own objective), and  
   - **Partial transport cost** among departments that already occupy cells (same flow matrix and rectilinear distances as CRAFT).

3. **CORELAP** (`corelap.py`)  
   Configures the module-level data used by `calculate_tcr`, `get_placement_order`, `get_neighbors`, `get_boundary_cells`, and `pr_weights`, then runs the same **Phase 2** placement loop as the script section in `corelap.py` (center seed, boundary expansion, PR maximization for each grid block). After each full department is placed, it records **partial transport cost** from current cell centroids.

4. **CRAFT** (`craft.py`)  
   Converts each construction layout to **department centroids** (centers of assigned cells in grid coordinates). Each centroid set is passed to `craft_pairwise_exchange` with the shared flow matrix and **all** department pairs as feasible swaps (`(i,j)` for `i < j`). Iterations stop when no improving swap exists or a layout cycle is detected (unchanged CRAFT logic).

5. **Reporting**  
   Console output includes ALDEP sequence, CORELAP placement order, transport cost **before** CRAFT for both constructions, and **final transport cost** after CRAFT for each seed (from `craft.py`). It also prints transport cost evaluated at **geometric centroids** of the **visual-replay** layout (see below); that number can differ from abstract CRAFT.

6. **Figures** (saved under `plots/`)  
   Layout figures (`01`–`04`, `11`) share one style: equal cells, **fixed color per department id** (Machining … Shipping always the same hue), labels show **name and current cell count** read from the grid, light grid lines, row 0 at the top, layout *y* flipped for Matplotlib.  

   **CRAFT layout plots** (`03`, `04`, lower row of `11`) do **not** change `craft.py`. They **replay each accepted CRAFT pair** on a copy of the construction grid using a **visual repartition rule**: if the two departments have the **same** number of cells, **swap** all their cells; if **unequal**, let *k* be the smaller department’s cell count, assign *k* cells taken from the **larger** department’s region that are **farthest** (Manhattan) from the smaller department’s cells (interior of the large region, away from the common boundary), label those *k* cells as the **smaller** id, and label every other cell in the union of the two as the **larger** id—so each department still occupies **k** and **(nₐ+nᵦ−k)** cells respectively. Rings and text mark **geometric centroids** of that replayed grid (same coordinate fix as before). Dashed lines / `+` appear only when optional abstract centroids are overlaid (not used for the final CRAFT-only view).  

   | File | Content |
   |------|---------|
   | `01_aldep_initial_layout.png` | ALDEP grid after construction |
   | `02_corelap_initial_layout.png` | CORELAP grid after construction |
   | `03_craft_final_from_aldep.png` | Visual replay after CRAFT (ALDEP seed) + geometric centroids |
   | `04_craft_final_from_corelap.png` | Visual replay after CRAFT (CORELAP seed) + geometric centroids |
   | `05_aldep_rel_score_vs_steps.png` | ALDEP REL score and partial transport vs departments placed |
   | `06_corelap_transport_vs_steps.png` | Partial transport vs CORELAP placement steps |
   | `07_craft_cost_vs_iteration_aldep_seed.png` | CRAFT total transport vs iteration (ALDEP seed) |
   | `08_craft_cost_vs_iteration_corelap_seed.png` | CRAFT total transport vs iteration (CORELAP seed) |
   | `09_craft_cost_both_seeds.png` | Both CRAFT curves on one chart |
   | `10_final_cost_comparison.png` | Bar chart of final costs after CRAFT |
   | `11_overview_four_panels.png` | Four-panel overview in the same unified layout style |

Run:

```bash
python main.py
```

Randomness in ALDEP uses `random.seed(42)` in `main.py` for reproducible sequences.

## Running the original modules alone

- `python aldep.py` — demo from the Colab export.  
- `python corelap.py` — demo with plots (`plt.show()`).  
- `python craft.py` — textbook-style five-department CRAFT example.

## Note on `corelap.py`

The interactive demo at the bottom of `corelap.py` is wrapped in `if __name__ == "__main__":` so that `import corelap` from `main.py` does not execute placement or open figure windows. The algorithm functions and constants are unchanged.

## CRAFT vs construction layouts

ALDEP and CORELAP assign **rectangular grid cells** to departments. **`craft.py` is unchanged**: CRAFT still optimizes by **swapping centroid coordinates** only. **`main.py`** additionally builds a **drawing-only** grid by replaying the same swap **pairs** with the **repartition rule** above so unequal areas stay consistent (*k* cells for the smaller department, *nₐ+nᵦ−k* for the larger). Bar and iteration charts still use abstract CRAFT costs from `craft_pairwise_exchange`.

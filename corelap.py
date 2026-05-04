# -*- coding: utf-8 -*-
"""FLD.ipynb


Original file is located at
    https://colab.research.google.com/drive/15KmnmxZg6qiSe79ZWnnT1dMmzBZK2ovJ
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. ALGORITHM CONFIGURATION ---
# Weights for Total Closeness Rating (TCR)
tcr_weights = {'A': 6, 'E': 5, 'I': 4, 'O': 3, 'U': 2, 'X': 1, '-': 0}

# Exponential weights for Placement Rating (PR)
pr_weights = {'A': 243, 'E': 81, 'I': 27, 'O': 9, 'U': 1, 'X': -729, '-': 0}

# --- 2. DUMMY PROBLEM DATA (VARYING SIZES) ---
departments = [0, 1, 2, 3, 4]
dept_names = ['Machining', 'Assembly', 'Tooling', 'Testing', 'Shipping']

# Sizes in grids: Machining(3), Assembly(2), Tooling(1), Testing(2), Shipping(1)
dept_sizes = {0: 3, 1: 2, 2: 1, 3: 2, 4: 1}

# REL-Chart
rel_chart = [
    ['-', 'A', 'E', 'I', 'O'],
    ['A', '-', 'U', 'X', 'I'],
    ['E', 'U', '-', 'A', 'E'],
    ['I', 'X', 'A', '-', 'U'],
    ['O', 'I', 'E', 'U', '-']
]

# --- 3. CORE FUNCTIONS ---
def calculate_tcr():
    """Calculates TCR for each department."""
    tcrs = {}
    for i in range(len(departments)):
        tcrs[i] = sum(tcr_weights[rel] for rel in rel_chart[i])
    return tcrs

def get_placement_order(tcrs):
    """Determines the placement sequence with tie-breakers (Area/Size)."""
    unplaced = list(departments)
    placed_order = []

    # 1. First Dept: Max TCR, tie-breaker: Max Area (size)
    first_dept = max(unplaced, key=lambda x: (tcrs[x], dept_sizes[x]))
    placed_order.append(first_dept)
    unplaced.remove(first_dept)

    hierarchy = ['A', 'E', 'I', 'O', 'U', 'X']

    # 2. Subsequent Depts
    while unplaced:
        best_candidate = None
        best_rel_index = 999

        for candidate in unplaced:
            # Strongest relation to ANY already placed department
            best_rel_for_candidate = 999
            for placed_dept in placed_order:
                rel = rel_chart[candidate][placed_dept]
                rel_idx = hierarchy.index(rel) if rel in hierarchy else 999
                if rel_idx < best_rel_for_candidate:
                    best_rel_for_candidate = rel_idx

            # Compare to find the winner
            if best_rel_for_candidate < best_rel_index:
                best_rel_index = best_rel_for_candidate
                best_candidate = candidate
            elif best_rel_for_candidate == best_rel_index:
                # Tie-breaker 1: TCR. Tie-breaker 2: Area
                if best_candidate is None:
                    best_candidate = candidate
                else:
                    cand_tup = (tcrs[candidate], dept_sizes[candidate])
                    best_tup = (tcrs[best_candidate], dept_sizes[best_candidate])
                    if cand_tup > best_tup:
                        best_candidate = candidate

        placed_order.append(best_candidate)
        unplaced.remove(best_candidate)

    return placed_order

def get_neighbors(grid, r, c):
    """Returns valid adjacent coordinates."""
    neighbors = []
    rows, cols = grid.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors

def get_boundary_cells(grid, restrict_to_dept=None):
    """Finds empty cells adjacent to the layout, or adjacent to a specific dept."""
    boundary = set()
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -1: # If cell is empty
                neighbors = get_neighbors(grid, r, c)
                for nr, nc in neighbors:
                    # If we are expanding a multi-grid dept, it must touch its own grids
                    if restrict_to_dept is not None:
                        if grid[nr, nc] == restrict_to_dept:
                            boundary.add((r, c))
                    # Otherwise, it just needs to touch the existing layout
                    elif grid[nr, nc] != -1:
                        boundary.add((r, c))
    return boundary

def plot_grid(grid, step_info):
    """Visualizes the grid state."""
    plt.figure(figsize=(6, 5))
    mask = grid == -1
    labels = np.empty_like(grid, dtype=object)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = grid[r, c]
            if val != -1:
                # Add size info to the label
                labels[r, c] = f"{dept_names[val]}\n(Size: {dept_sizes[val]})"
            else:
                labels[r, c] = ""

    sns.heatmap(grid, annot=labels, fmt="", cmap="Pastel1", mask=mask,
                cbar=False, linewidths=2, linecolor='black')
    plt.title(step_info, pad=20, fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.show()

# --- 4. EXECUTING CORELAP WITH MULTI-GRIDS ---
if __name__ == "__main__":
    print("--- CORELAP Phase 1: Placement Order ---")
    tcrs = calculate_tcr()
    for d, tcr in tcrs.items():
        print(f"{dept_names[d]} (Size {dept_sizes[d]}) - TCR: {tcr}")

    placement_order = get_placement_order(tcrs)
    print("\nFinal Placement Order:", [dept_names[d] for d in placement_order])
    print("\n--- CORELAP Phase 2: Layout Placement ---")

    # Grid needs to be larger to accommodate multi-grid departments comfortably
    grid_size = 7
    layout_grid = np.full((grid_size, grid_size), -1)

    for step, dept in enumerate(placement_order):
        size_needed = dept_sizes[dept]

        for block_num in range(size_needed):
            if step == 0 and block_num == 0:
                # First block of first department goes in the center
                center = grid_size // 2
                layout_grid[center, center] = dept
                continue

            # Determine valid locations for this specific block
            if block_num == 0:
                # First block of a new dept must attach to the existing layout boundary
                candidate_locations = get_boundary_cells(layout_grid)
            else:
                # Subsequent blocks of the SAME dept must attach to its own currently placed blocks
                candidate_locations = get_boundary_cells(layout_grid, restrict_to_dept=dept)

            # Evaluate Placement Rating (PR)
            best_loc = None
            max_pr = -float('inf')

            for r, c in candidate_locations:
                pr_score = 0
                neighbors = get_neighbors(layout_grid, r, c)
                for nr, nc in neighbors:
                    adj_dept = layout_grid[nr, nc]
                    # Calculate PR based on relationships with *other* departments
                    if adj_dept != -1 and adj_dept != dept:
                        rel = rel_chart[dept][adj_dept]
                        pr_score += pr_weights[rel]

                if pr_score > max_pr:
                    max_pr = pr_score
                    best_loc = (r, c)

            # Place the block
            layout_grid[best_loc[0], best_loc[1]] = dept

        # Plot only after the ENTIRE department (all its grids) is placed
        plot_grid(layout_grid, f"Step {step+1}: Placed '{dept_names[dept]}' ({size_needed} grids)")

    print("Layout placement complete!")


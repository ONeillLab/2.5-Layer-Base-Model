import numpy as np

def get_surrounding_points_with_wrapping(arr, i, j):
    # List of relative indices for surrounding points (including diagonals)
    surrounding_indices = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1), (1, 0), (1, 1)]
    
    rows, cols = arr.shape
    surrounding_points = []
    
    for di, dj in surrounding_indices:
        ni, nj = (i + di) % rows, (j + dj) % cols
        surrounding_points.append((ni, nj, arr[ni, nj]))
    
    return surrounding_points

# Example usage
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
i, j = 0, 0  # Point of interest (top-left corner)

surrounding_points = get_surrounding_points_with_wrapping(arr, i, j)
print("Surrounding points with wrapping:", surrounding_points)
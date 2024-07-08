import numpy as np
import time

timer = time.time()
def get_surrounding_points_2_away_with_wrapping(arr, i, j, n):
    rows, cols = arr.shape
    
    # Initialize an (n+4)x(n+4) result array
    result = np.full((n+4, n+4), np.nan, dtype=arr.dtype)
    
    #Fill the center (n x n) part with the original n x n block
    for block_i in range(n):
        for block_j in range(n):
            result[block_i + 2, block_j + 2] = arr[(i + block_i) % rows, (j + block_j) % cols]
    
    #result[2:n+2,:][:,2:n+2] = arr[i:i+n,:][:,j:j+n]

    # Fill the edges 2 cells away
    for block_j in range(n):
        result[0, block_j + 2] = arr[(i - 2) % rows, (j + block_j) % cols]  # Top edge
        result[1, block_j + 2] = arr[(i - 1) % rows, (j + block_j) % cols]  # 1 cell above the block
        result[n + 2, block_j + 2] = arr[(i + n) % rows, (j + block_j) % cols]  # 1 cell below the block
        result[n + 3, block_j + 2] = arr[(i + n + 1) % rows, (j + block_j) % cols]  # Bottom edge
    
    for block_i in range(n):
        result[block_i + 2, 0] = arr[(i + block_i) % rows, (j - 2) % cols]  # Left edge
        result[block_i + 2, 1] = arr[(i + block_i) % rows, (j - 1) % cols]  # 1 cell left of the block
        result[block_i + 2, n + 2] = arr[(i + block_i) % rows, (j + n) % cols]  # 1 cell right of the block
        result[block_i + 2, n + 3] = arr[(i + block_i) % rows, (j + n + 1) % cols]  # Right edge
    
    # Fill the corners 2 cells away
    result[0, 0] = arr[(i - 2) % rows, (j - 2) % cols]  # Top-left corner
    result[0, 1] = arr[(i - 2) % rows, (j - 1) % cols]  # Top-left 1 cell right
    result[1, 0] = arr[(i - 1) % rows, (j - 2) % cols]  # Top-left 1 cell down
    result[1, 1] = arr[(i - 1) % rows, (j - 1) % cols]  # Top-left 1 cell down-right

    result[0, n + 3] = arr[(i - 2) % rows, (j + n + 1) % cols]  # Top-right corner
    result[0, n + 2] = arr[(i - 2) % rows, (j + n) % cols]  # Top-right 1 cell left
    result[1, n + 3] = arr[(i - 1) % rows, (j + n + 1) % cols]  # Top-right 1 cell down
    result[1, n + 2] = arr[(i - 1) % rows, (j + n) % cols]  # Top-right 1 cell down-left

    result[n + 3, 0] = arr[(i + n + 1) % rows, (j - 2) % cols]  # Bottom-left corner
    result[n + 3, 1] = arr[(i + n + 1) % rows, (j - 1) % cols]  # Bottom-left 1 cell right
    result[n + 2, 0] = arr[(i + n) % rows, (j - 2) % cols]  # Bottom-left 1 cell up
    result[n + 2, 1] = arr[(i + n) % rows, (j - 1) % cols]  # Bottom-left 1 cell up-right

    result[n + 3, n + 3] = arr[(i + n + 1) % rows, (j + n + 1) % cols]  # Bottom-right corner
    result[n + 3, n + 2] = arr[(i + n + 1) % rows, (j + n) % cols]  # Bottom-right 1 cell left
    result[n + 2, n + 3] = arr[(i + n) % rows, (j + n + 1) % cols]  # Bottom-right 1 cell up
    result[n + 2, n + 2] = arr[(i + n) % rows, (j + n) % cols]  # Bottom-right 1 cell up-left
    
    return result

# Example usage
arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]])

i, j = 3,2  # Top-left corner of the n x n block
n = 2  # Size of the block

result_array = get_surrounding_points_2_away_with_wrapping(arr, i, j, n)
print(f"Result in time {time.time()-timer}")
print(result_array)
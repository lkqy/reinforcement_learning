def yakebi(alpha, policy):
    matrix = [[0 for _ in range(4)] for _ in range(4)]
    tmp_matrix = [[0 for _ in range(4)] for _ in range(4)]

    for k in range(30):
        for i in range(4):
            for j in range(4):
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    continue
                tmp_matrix[i][j] = policy(matrix, i, j)

        flag = True
        for i in range(4):
            for j in range(4):
                if abs(matrix[i][j]-tmp_matrix[i][j]) > alpha:
                    flag = False
                matrix[i][j] = tmp_matrix[i][j]

        
    
    print 'yakebi', k
    for i in range(4):
        print matrix[i]

def gassi(alpha, policy):
    matrix = [[0 for _ in range(4)] for _ in range(4)]

    for k in range(30):
        flag = True
        for i in range(4):
            for j in range(4):
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    continue
                matrix[i][j] = policy(matrix, i, j)

        
    print 'gassi', k
    for i in range(4):
        print matrix[i]

def avg_policy(matrix, i, j):
    left = matrix[i][j-1] if j - 1 >= 0 else -1
    right = matrix[i][j+1] if j + 1 < 4 else -1
    top = matrix[i-1][j] if i - 1 >= 0 else -1
    down = matrix[i+1][j] if i + 1 < 4 else -1
    tmp = -1 + 0.25 * (left + right + top + down)
    return tmp

def max_policy(matrix, i, j):
    left = matrix[i][j-1] if j - 1 >= 0 else -1
    right = matrix[i][j+1] if j + 1 < 4 else -1
    top = matrix[i-1][j] if i - 1 >= 0 else -1
    down = matrix[i+1][j] if i + 1 < 4 else -1
    return -1 + max(left, right, top, down)

alpha=0.00000000000000001
yakebi(alpha, max_policy)
gassi(alpha, max_policy)

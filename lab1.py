import numpy as np

def read_lp(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    task_type = lines[0]  # max / min
    c = np.array([float(x) for x in lines[1].split()])
    m = int(lines[2])

    A, b, signs = [], [], []
    for line in lines[3:3 + m]:
        parts = line.split()
        A.append([float(x) for x in parts[:-2]])
        signs.append(parts[-2])
        b.append(float(parts[-1]))

    return task_type, np.array(A), np.array(b), c, signs


def to_canonical(A, b, signs):
    """Приведение к каноническому виду: <= → +s, >= → -s"""
    m, n = A.shape
    slack_vars = []
    A_canon = []
    for i in range(m):
        row = list(A[i])
        if signs[i] == "<=":
            row += [1 if j == i else 0 for j in range(m)]
        elif signs[i] == ">=":
            row += [-1 if j == i else 0 for j in range(m)]
        else:  # "="
            row += [0 for _ in range(m)]
        A_canon.append(row)
    return np.array(A_canon, dtype=float), b


def simplex(A, b, c):
    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    basis = list(range(n - m, n))

    while True:
        col = np.argmin(tableau[-1, :-1])
        if tableau[-1, col] >= 0:
            break

        ratios = []
        for i in range(m):
            if tableau[i, col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, col])
            else:
                ratios.append(np.inf)
        row = np.argmin(ratios)
        if ratios[row] == np.inf:
            print("Функция не ограничена сверху.")
            return None, None

        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(m + 1):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

        basis[row] = col

    x = np.zeros(n)
    for i in range(m):
        x[basis[i]] = tableau[i, -1]

    z = tableau[-1, -1]
    return x, z


# --- Основная часть ---
task_type, A, b, c, signs = read_lp("zlp.txt")
A_canon, b = to_canonical(A, b, signs)

# Добавляем нули в c для новых переменных
extra_vars = A_canon.shape[1] - len(c)
c_full = np.concatenate([c, np.zeros(extra_vars)])

if task_type == "min":
    c_full = -c_full

x_opt, z_opt = simplex(A_canon, b, c_full)

if x_opt is not None:
    print("Оптимальное решение найдено:")
    for i, val in enumerate(x_opt, start=1):
        print(f"x{i} = {val:.2f}")
    print(f"Zmax = {z_opt:.2f}")
else:
    print("Решений нет или функция не ограничена.")


import re
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def parse_expression(expr):
    expr = expr.replace(" ", "")
    terms = re.findall(r'([+-]?\d*\.?\d*)x(\d+)', expr)
    max_var = max(int(v) for _, v in terms)
    coeffs = [0] * max_var
    for coef, var in terms:
        coef = float(coef) if coef not in ("", "+", "-") else (1.0 if coef != "-" else -1.0)
        coeffs[int(var) - 1] = coef
    return coeffs

def read_lpp_expressions(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    func_line = lines[0]
    mode = "max" if "max" in func_line.lower() else "min"
    func_expr = func_line.split("->")[0].strip()
    c = np.array(parse_expression(func_expr))

    A, b, signs = [], [], []
    for line in lines[1:]:
        match = re.split(r'(<=|>=|=)', line)
        left, sign, right = match[0].strip(), match[1], match[2].strip()
        A.append(parse_expression(left))
        b.append(float(right))
        signs.append(sign)

    if mode == "min":
        c = -c

    return np.array(A, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float), signs

def canonical_form(A, b, c, signs):
    m, n = A.shape
    A_new = []
    for i in range(m):
        row = list(A[i])
        if signs[i] == "<=":
            slack = [0] * m
            slack[i] = 1
            row.extend(slack)
        elif signs[i] == ">=":
            slack = [0] * m
            slack[i] = -1
            row.extend(slack)
        elif signs[i] == "=":
            slack = [0] * m
            row.extend(slack)
        A_new.append(row)

    A_new = np.array(A_new, dtype=float)
    c_new = np.concatenate([c, np.zeros(A_new.shape[1] - len(c))])
    return A_new, np.array(b), c_new

def print_table(table, basis, iteration):
    m, n = table.shape
    print(f"\n=== Итерация {iteration} ===")
    header = ["Базис"] + [f"x{i+1}" for i in range(n - 1)] + ["b"]
    print(" | ".join(f"{h:^8}" for h in header))
    print("-" * (10 * (n + 1)))
    for i in range(m - 1):
        row = [f"x{basis[i]+1}"] + [f"{table[i, j]:8.3f}" for j in range(n)]
        print(" | ".join(row))
    print("-" * (10 * (n + 1)))
    row = ["F"] + [f"{table[-1, j]:8.3f}" for j in range(n)]
    print(" | ".join(row))
    print()

def simplex(A, b, c):
    m, n = A.shape
    A = np.hstack([A, np.eye(m)])
    c = np.concatenate([c, np.zeros(m)])
    table = np.zeros((m + 1, n + m + 1))
    table[:-1, :-1] = A
    table[:-1, -1] = b
    table[-1, :-1] = -c
    basis = list(range(n, n + m))
    iteration = 0
    print_table(table, basis, iteration)

    while True:
        iteration += 1
        if np.all(table[-1, :-1] >= -1e-9):
            break
        pivot_col = np.argmin(table[-1, :-1])
        if np.all(table[:-1, pivot_col] <= 0):
            print("Целевая функция не ограничена!")
            return None, None
        ratios = [
            table[i, -1] / table[i, pivot_col] if table[i, pivot_col] > 0 else np.inf
            for i in range(m)
        ]
        pivot_row = np.argmin(ratios)
        pivot = table[pivot_row, pivot_col]
        table[pivot_row, :] /= pivot
        for i in range(m + 1):
            if i != pivot_row:
                table[i, :] -= table[i, pivot_col] * table[pivot_row, :]
        basis[pivot_row] = pivot_col
        print(f" Входит x{pivot_col+1}, выходит x{basis[pivot_row]+1}")
        print_table(table, basis, iteration)
        if iteration > 50:
            print("Превышено число итераций (возможен цикл).")
            break

    x = np.zeros(n + m)
    for i in range(m):
        x[basis[i]] = table[i, -1]
    return x[:n], table[-1, -1]

def solve_expressions(filename):
    A, b, c, signs = read_lpp_expressions(filename)
    A, b, c = canonical_form(A, b, c, signs)
    x, f = simplex(A, b, c)
    if x is not None:
        print("\n=== РЕЗУЛЬТАТ ===")

        num_vars = len(c) - len(A)
        x_main = x[:num_vars]

        for i, val in enumerate(x_main, start=1):
            print(f"x{i} = {round(val, 3)}")

        print(f"F = {round(f, 3)}")

    else:
        print("\n Решение не найдено.")

    return x, f

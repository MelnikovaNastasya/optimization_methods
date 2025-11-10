import numpy as np
from scipy.optimize import linprog

def read_lp(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    task_type = lines[0].lower()  # max / min
    c = np.array([float(x) for x in lines[1].split()])
    m = int(lines[2])

    A, b, signs = [], [], []
    for line in lines[3:3 + m]:
        parts = line.split()
        A.append([float(x) for x in parts[:-2]])
        signs.append(parts[-2])
        b.append(float(parts[-1]))

    bounds = [(0, None)] * len(c)
    if len(lines) > 3 + m and lines[3 + m].lower().startswith("bounds"):
        bounds_info = lines[4 + m].split()
        bounds = [(0, None) if bnd == "0" else (None, None) for bnd in bounds_info]

    return task_type, np.array(A, float), np.array(b, float), c, signs, bounds


def solve_lp(filename):
    task_type, A, b, c, signs, bounds = read_lp(filename)

    print("СЧИТАННАЯ ЗАДАЧА")
    print("Тип задачи:", "максимизация" if task_type == "max" else "минимизация")
    print("Целевая функция: ", c)
    print("Ограничения:")
    for i, s in enumerate(signs):
        print(f"  {A[i]} {s} {b[i]}")
    print("Границы переменных:", bounds)

    # Приведение к каноническому виду
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []
    for i, s in enumerate(signs):
        if s == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif s == ">=":
            A_ub.append(-A[i])
            b_ub.append(-b[i])
        elif s == "=":
            A_eq.append(A[i])
            b_eq.append(b[i])

    # Формирование целевой функции
    c_obj = -c if task_type == "max" else c

    # Решение задачи линейного программирования
    res = linprog(
        c=c_obj,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs"
    )

    print("РЕШЕНИЕ")
    if res.success:
        print("→ Оптимальное решение найдено:")
        for i, val in enumerate(res.x, 1):
            print(f"x{i} = {val:.4f}")
        z_val = -res.fun if task_type == "max" else res.fun
        print(f"Z* = {z_val:.4f}")
    else:
        print("→ Решение не найдено:", res.message)



if __name__ == "__main__":
    solve_lp("task.txt")













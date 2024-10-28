import numpy as np


def p1a():
    A = np.array([[3, -1], [-1.5, 7]])
    B = np.array([7.5, 7])
    X = np.linalg.solve(A, B)
    print(X)
    Ai = np.linalg.inv(A)
    X2 = Ai @ B
    print(X2)


def p1b():
    A = np.array([[3, -1, -2], [-1.5, 7, 1]])
    B = np.array([3.5, 9])
    At = A.T
    a = At @ A
    b = At @ B
    X = np.linalg.solve(a, b)
    B2 = A @ X
    print(X)
    print(B2)


def p1c():
    A = np.array([[3, -1, -2], [-1.5, 7, 1]])
    B = np.array([3.5, 9])
    At = A.T
    a = At @ A
    ai = np.linalg.inv(a)
    b = At @ B
    X = np.linalg.solve(ai, b)
    B2 = A @ X
    print(X)
    print(B2)


def main():
    print("Problem 1a")
    p1a()
    print("Problem 1b")
    p1b()
    print("Problem 1c")
    p1c()


if __name__ == "__main__":
    main()
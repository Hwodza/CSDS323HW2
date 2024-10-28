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


def p3a():
    P = np.array([[1, 2, -1], [2, 1, 2], [-1, 2, 1]])

    u = np.array([[0], [1], [1]])
    v = np.array([[1], [0], [1]])

    # Compute Q = P + u * v^T
    Q = P + np.dot(u, v.T)

    # Compute the inverse of P and Q using numpy.linalg.inv()
    P_inv = np.linalg.inv(P)
    Q_inv = np.linalg.inv(Q)

    # Check if Q * Q_inv is close to the identity matrix
    identity_check = np.allclose(np.dot(Q, Q_inv), np.eye(3))
    PPi = np.matmul(P, P_inv)
    QQi = np.matmul(Q, Q_inv)
    print(PPi)
    print(QQi)
    print()
    print(P_inv, Q_inv, identity_check)


def p3b():
    P = np.array([[1, 2, -1], [2, 1, 2], [-1, 2, 1]])
    u = np.array([[0], [1], [1]])
    v = np.array([[1], [0], [1]])

    # Compute Q using the rank-1 update: Q = P + u * v^T
    Q = P + np.dot(u, v.T)
    # Direct inversion of Q using numpy's built-in function
    Q_inv_direct = np.linalg.inv(Q)
    # Compute Q inverse using the Sherman-Morrison formula:
    # Q_inv = P_inv - (P_inv * u * v^T * P_inv) / (1 + v^T * P_inv * u)
    # Step 1: Compute the inverse of P
    P_inv = np.linalg.inv(P)

    # Step 2: Compute the scalar (1 + v^T * P_inv * u)
    vT_Pinv_u = np.dot(np.dot(v.T, P_inv), u)
    denominator = 1 + vT_Pinv_u

    # Step 3: Compute the rank-1 update term: (P_inv * u * v^T * P_inv) / denominator
    numerator = np.dot(P_inv, np.dot(u, np.dot(v.T, P_inv)))
    Q_inv_sherman_morrison = P_inv - numerator / denominator
    QQi = np.matmul(Q, Q_inv_direct)
    print(QQi)
    # Verify if Q_inv_sherman_morrison is close to Q_inv_direct by checking their difference
    difference = np.allclose(Q_inv_direct, Q_inv_sherman_morrison)
    Q_inv_direct, Q_inv_sherman_morrison, difference


def main():
    print("Problem 1a")
    p1a()
    print("Problem 1b")
    p1b()
    print("Problem 1c")
    p1c()
    print("Problem 3a")
    p3a()
    print("Problem 3b")
    p3b()


if __name__ == "__main__":
    main()
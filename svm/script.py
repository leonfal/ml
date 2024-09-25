import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def kernel(x: np.array, y: np.array, type: str = "rbf") -> float:
    """Function to calculate the kernel of two vectors

    Args:
        x (np.array): first vector
        y (np.array): second vector
        type (str): type of kernel to be used
        parameter (int): parameter for the kernel function

    Returns:
        float: kernel value
    """

    if type == "linear":
        return np.dot(x, y)

    elif type == "polynomial":
        parameter = 2
        return np.power((np.dot(x, y) + 1), parameter)

    elif type == "rbf":
        sigma = 1.6
        return math.exp(
            -math.pow(np.linalg.norm(np.subtract(x, y)), 2) / (2 * math.pow(sigma, 2))
        )


# dual formulation
def objective(alpha: np.array) -> float:
    """Function to return scalar objective from alpha vector input"""

    double_sum = 0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            double_sum += alpha[i] * alpha[j] * P[i][j]

    return 0.5 * double_sum - np.sum(alpha)


def zerofun(alpha: np.array) -> float:
    """Function to return scalar constraint from alpha vector input"""
    return sum(alpha[i] * targets[i] for i in range(N))


def indicator(
    alpha: np.array, true: np.array, x: np.array, s: np.array, b: np.float64
) -> float:
    """ "Function to return scalar indicator from alpha vector input

    Args:
        alpha: alpha vector
        true: true labels of x
        x: non zero values of alpha (support vectors "support")
        s: what we want to classify
        b: bias term

    Returns:
        float: indicator value
    """
    # indicator = np.sum(alpha * true * kernel(s, x)) - b

    indicator_sum = 0
    for i in range(len(alpha)):
        indicator_sum += alpha[i] * true[i] * kernel(s, x[i])

    return indicator_sum - b


classa = np.concatenate(
    (
        np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
        np.random.randn(10, 2) * 0.2 + [-1.5, 0.5],
    )
)
classb = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
C = 10

inputs = np.concatenate((classa, classb))
targets = np.concatenate((np.ones(classa.shape[0]), -np.ones(classb.shape[0])))

plt.scatter(classa[:, 0], classa[:, 1], color="blue", marker="o", label="Class A")
plt.scatter(classb[:, 0], classb[:, 1], color="red", marker="o", label="Class B")
plt.legend()
plt.show()

N = inputs.shape[0]  # Number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]
start = np.zeros(N)  # initial guess

P = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        P[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])


def main():
    ret = minimize(
        objective,
        start,
        bounds=[(0, C) for b in range(N)],
        constraints={"type": "eq", "fun": zerofun},
    )

    alpha = ret["x"]

    alpha_list = []
    support_list = []
    true_list = []
    # Find the support vectors
    for i in range(len(alpha)):
        if alpha[i] > 10**-5:
            alpha_list.append(alpha[i])  # alpha non zero values
            support_list.append(inputs[i])  # associated support vectors
            true_list.append(targets[i])  # associated true labels of support vectors

    choosen_support_vector = 0

    # convert lists to NumPy arrays
    support_vectors_data = {
        "alpha": np.array(alpha_list),
        "support": np.array(support_list),
        "true": np.array(true_list),
    }

    non_zero_sum = 0
    for i in range(len(support_vectors_data["alpha"])):
        non_zero_sum += (
            support_vectors_data["alpha"][i]
            * support_vectors_data["true"][i]
            * kernel(
                support_vectors_data["support"][choosen_support_vector],
                support_vectors_data["support"][i],
            )
        )

    b = non_zero_sum - support_vectors_data["true"][choosen_support_vector]

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array(
        [
            [
                indicator(
                    support_vectors_data["alpha"],
                    support_vectors_data["true"],
                    support_vectors_data["support"],
                    np.array([x, y]),  # s
                    b,
                )
                for x in xgrid
            ]
            for y in ygrid
        ]
    )

    plt.plot(
        [p[0] for p in classa],
        [p[1] for p in classa],
        ".",
        color="green",
        label="Class A",
    )
    plt.plot(
        [p[0] for p in classb],
        [p[1] for p in classb],
        ".",
        color="purple",
        label="Class B",
    )
    plt.contour(
        xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=("purple", "black", "green")
    )
    plt.axis("equal")
    plt.savefig("plots/svmplot-rbf.pdf")
    # name x axis and y axis
    plt.xlabel("x")
    plt.ylabel("y")
    # add legend
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

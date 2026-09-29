def finite_difference(function, x, epsilon=1e-4):
    """用中心有限差分估计一元函数在 x 处的导数。"""
    left = function(x - epsilon)
    right = function(x + epsilon)
    return (right - left) / (2 * epsilon)


def cubic(x):
    return x**3 + 2 * x


def loss_for_weight(weight):
    x = 2.0
    target = 5.0
    bias = 0.0
    prediction = weight * x + bias
    return (prediction - target) ** 2


if __name__ == "__main__":
    print("f(x) = x^3 + 2x，在 x = 2 处：")
    for epsilon in (1e-1, 1e-2, 1e-4):
        estimate = finite_difference(cubic, 2.0, epsilon)
        print(f"  epsilon={epsilon:g}, 导数估计={estimate:.6f}")

    weight = 1.0
    epsilon = 1e-4
    learning_rate = 0.1
    before = loss_for_weight(weight)
    gradient = finite_difference(loss_for_weight, weight, epsilon)
    updated_weight = weight - learning_rate * gradient
    after = loss_for_weight(updated_weight)

    print("\n权重更新：")
    print(f"  更新前: weight={weight:.6f}, loss={before:.6f}")
    print(f"  有限差分梯度: dL/dw={gradient:.6f}")
    print(f"  更新后: weight={updated_weight:.6f}, loss={after:.6f}")

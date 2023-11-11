import random

from scipy.special._ufuncs import expit

from extend_functions import *


def square(x):
    return np.power(x, 2)


def cube(x):
    return np.power(x, 3)


class Operator:
    def __init__(self, operation, parity, string, infix, infix_name, weight=1):
        self.operation = operation
        self.parity = parity
        self.string = string
        self.infix = infix
        self.infix_name = infix_name
        self.weight = weight


class OperatorDistribution:
    def __init__(self):
        self.operators_map = {}
        self.weights = []
        self.weights_are_current = False

    def add(self, operator):
        self.operators_map[operator.infix_name] = operator
        self.weights_are_current = False

    def get_random(self, k=1):
        if not self.weights_are_current:
            self.weights = list(
                map(lambda x: self.operators_map[x].weight, self.operators_map)
            )
            self.weights_are_current = True
        return random.choices(
            list(self.operators_map.keys()), weights=self.weights, k=k
        )

    def get(self, key):
        return self.operators_map[key]

    def contains(self, key):
        return key in self.operators_map.keys()


default_operators = OperatorDistribution()
ops = [
    Operator(np.add, 2, "({0} + {1})", "add({0},{1})", "add"),
    Operator(np.subtract, 2, "({0} - {1})", "sub({0},{1})", "sub"),
    Operator(np.multiply, 2, "({0} * {1})", "mul({0},{1})", "mul"),
    Operator(protect_divide, 2, "({0} / {1})", "div({0},{1})", "div"),
    Operator(None, None, None, None, "mutate"),
    Operator(None, None, None, None, "transition"),
]
optional_ops = [
    Operator(np.negative, 1, "(-{0})", "neg({0})", "neg"),
    Operator(protect_mod, 2, "({0} % {1})", "mod({0},{1})", "mod"),
    Operator(np_max, 2, "max({0},{1})", "max({0},{1})", "max"),
    Operator(np_min, 2, "min({0},{1})", "min({0},{1})", "min"),
    Operator(expit, 1, "sigmoid({0})", "sigmoid({0})", "sigmoid"),
    Operator(protect_loge, 1, "log({0})", "log({0})", "log"),
    Operator(protect_log2, 1, "log2({0})", "log2({0})", "log2"),
    Operator(protect_log10, 1, "log10({0})", "log10({0})", "log10"),
    Operator(square, 1, "sqr({0})", "sqr({0})", "sqr"),
    Operator(cube, 1, "cbe({0})", "cbe({0})", "cbe"),
    Operator(protect_sqrt, 1, "sqt({0})", "sqt({0})", "sqt"),
    Operator(np.cbrt, 1, "cbt({0})", "cbt({0})", "cbt"),
    Operator(np.sin, 1, "sin({0})", "sin({0})", "sin"),
    Operator(np.cos, 1, "cos({0})", "cos({0})", "cos"),
    Operator(np.abs, 1, "abs({0})", "abs({0})", "abs"),
]


def get_ops(gene):
    ops = []
    for i in range(len(optional_ops)):
        if gene[i] == 1:
            ops.append(optional_ops[i])
    return ops


cart_pole_optional_ops = get_ops([0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

mountain_car_optional_ops = get_ops([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])

acrobot_ops = get_ops([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0])

industrial_benchmark_ops = get_ops([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1])

for op in ops:
    default_operators.add(op)


def print_ops(op_ops, env_name):
    print(env_name)
    for o in ops + op_ops:
        infix = o.infix_name
        if infix == "mutate":
            continue
        if infix == "transition":
            continue
        if infix in infix_map.keys():
            print(infix_map[infix], end=",")
        else:
            print(infix, end=",")
    print()


if __name__ == "__main__":
    infix_map = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "mod": "\%",
        "log10": "log_{10}{x}",
        "log": "ln{x}",
        "log2": "log_{2}{x}",
        "sqr": "\sqr{x}",
        "sqt": "\sqrt{x}",
        "sin": "sin(x)",
        "cos": "cos(x)",
        "sigmoid": "logistic(x)",
        "abs": "|x|",
        "cbe": "x^3",
        "cbt": "\sqrt[3]{x}",
    }

    print_ops(cart_pole_optional_ops, "Cartpole")
    print_ops(mountain_car_optional_ops, "MountainCar")
    print_ops(acrobot_ops, "Acrobot")

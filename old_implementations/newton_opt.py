import numpy as np
from sympy import symbols, Eq, solve, log

def partial(element, function):
    partial_diff = function.diff(element)
    return partial_diff

def gradient(partials):
    grad = np.matrix([[partials[0]], [partials[1]]])
    return grad

def gradient_to_zero(symbols_list, partials):
    partial_x = Eq(partials[0], 0)
    partial_y = Eq(partials[1], 0)

    singular = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1]))
    return singular

def hessian(partials_second, cross_derivatives):
    hessianmat = np.matrix([[partials_second[0], cross_derivatives], [cross_derivatives, partials_second[1]]])
    return hessianmat

def determat(partials_second, cross_derivatives, singular, symbols_list):
    
    det = partials_second[0].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) * partials_second[1].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) - (cross_derivatives.subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]))**2

    return det

def main():
    x, y = symbols('x y')
    symbols_list = [x, y]
    function = x**2 - (3/2)*x*y + y**2

    partials, partials_second = [], []

    for element in symbols_list:
        partial_diff = partial(element, function)
        partials.append(partial_diff)

    grad = gradient(partials)
    singular = gradient_to_zero(symbols_list, partials)

    cross_derivatives = partial(symbols_list[0], partials[1])

    for i in range(0, len(symbols_list)):
        partial_diff = partial(symbols_list[i], partials[i])
        partials_second.append(partial_diff)

    hessianmat = hessian(partials_second, cross_derivatives)
    det = determat(partials_second, cross_derivatives, singular, symbols_list)

    print("Hessian is:", hessianmat)
    print("Determat is:", det)

main()


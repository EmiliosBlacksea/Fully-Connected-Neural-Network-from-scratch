# import numpy as np
# from functions.neural_network.neural import neural_layer

# def test():
#     Layer1 = neural_layer(2, 5, 5)
#     Layer2 = neural_layer(5, 1, None, None, False, False, True)

#     error = 100
#     while error > 10**(-5):
#     input = random_xy_scalar()
#     forward1 = Layer1.forward(input)
#     output = Layer2.forward(forward1)
#     error  = (FF(input) - output) ** 2
#     de_dy = -2 * error * output
#     grad1 = Layer2.backward(de_dy)
#     grad = Layer1.backward(grad1)
#     print(error)


#     print()

# def FF(input):
#     return input[0][0]**2 + 2 * input[0][1] + 1

# def random_xy_scalar(low=-4.0, high=4.0, seed=None):
#     rng = np.random.default_rng(seed)
#     x = float(rng.uniform(low, high))
#     y = float(rng.uniform(low, high))
#     return np.array([[x, y]], dtype=np.float32)
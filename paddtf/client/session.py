import paddle

class Session():
    def __init__(self):
        print("Session not implemented")
class linalg():
    def __init__(self):
        print("linalg init")
    def LinearOperatorLowerTriangular(self,input):
        return paddle.tensor.tril(input)

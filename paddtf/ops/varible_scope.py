import paddle

def get_variable(name, shape):
    return paddle.fluid.layers.create_parameter(shape=shape, dtype='float32', name=name)



def Variable(value=0, name=None, trainable=False):
    return paddle.fluid.layers.create_parameter(shape=[1], dtype='float32', name='fc_b')
                                           
def global_variables_initializer():
    print("global_variables_initializer not implemented")


def local_variables_initializer():
    print("local_variables_initializer not implemented")


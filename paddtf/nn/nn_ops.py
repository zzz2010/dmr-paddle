import paddle

def softmax(x):
    return paddle.nn.functional.softmax(x, axis=- 1, dtype=None, name=None)


def sampled_softmax_loss(weights,biases,labels,inputs,num_sampled=0,num_classes=None,sampled_values=None):
    outputs=paddle.matmul(inputs,weights,transpose_y=True)+paddle.reshape(biases,(1,-1))
    return paddle.nn.functional.softmax_with_cross_entropy(logits=outputs,
                                                           label=labels,
                                                           soft_label=False,
                                                           ignore_index=- 100,
                                                           numeric_stable_mode=True,
                                                           return_softmax=False,
                                                           axis=- 1)
def learned_unigram_candidate_sampler(true_classes,num_true,num_sampled,unique,range_max,seed=None,name=None):
    print("learned_unigram_candidate_sampler not implement")
    return None


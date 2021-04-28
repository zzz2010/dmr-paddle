import paddle

def embedding_lookup(params,ids):
    return paddle.gather(x=params,index=ids)
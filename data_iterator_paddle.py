import numpy as np
import gzip
import paddle


def reader_data(source, batch_size=256, max_batch_size=20):
    if source.endswith(".gz"):
        source = gzip.open(source, 'rb')
    else:
        source = open(source, 'r')
    source_buffer = []
    if len(source_buffer) == 0:
        for k_ in range(batch_size * max_batch_size):
            ss = source.readline()
            if not isinstance(ss, str):
                ss = ss.decode("utf-8")
            if ss == "":
                break
            source_buffer.append(ss.strip("\n").split(","))

    def reader():
        while len(source_buffer)>0:
            source = []
            target = []
            for _ in range(batch_size):
                try:
                    ss = source_buffer.pop()
                except IndexError:
                    break
                source.append(ss[:-1])
                target.append(ss[-1])
            source = np.array(source, np.float32)
            target = np.array(target, np.float32)
            yield source, target

    return reader

if __name__=="__main__":
    loader = paddle.io.DataLoader.from_generator(capacity=5)
    loader.set_batch_generator(reader_data())

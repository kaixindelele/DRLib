import numpy as np


class StateNorm:
    def __init__(self, size, eps=1e-2, default_clip_range=5):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.zeros(1, np.float32)

        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]

        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(np.square(self.eps),
                                      (self.sumsq / self.count) - np.square(
            self.sum / self.count)))
        # print("mean:", self.mean)
        # print("std:", self.std)

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range

        return np.clip((v - self.mean) / self.std,
                       -clip_range, clip_range)


def main():
    norm = Norm(size=3)
    v = np.random.random((4, 2, 3))
    print("v:", v)

    r0 = v.reshape(-1, 3)
    print(r0.shape)
    print(r0)
    r0 = r0[:, 0]

    print(r0.shape)
    print(r0)
    std = np.std(r0)
    print(std.shape)
    print(std)
    norm.update(v=v)


if __name__ == '__main__':
    main()

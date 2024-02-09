import numpy as np
class Utility:
    @staticmethod
    def convert_to_display(samples):
        cnt, height, width = int(np.floor(np.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        samples = np.reshape(samples, [height, cnt, cnt, width])
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        samples = np.reshape(samples, [height * cnt, width * cnt])
        return samples

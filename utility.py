import numpy as np
import os


class Utility:
    @staticmethod
    def convert_to_display(samples):
        cnt, height, width = int(np.floor(np.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        samples = np.reshape(samples, [height, cnt, cnt, width])
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        samples = np.reshape(samples, [height * cnt, width * cnt])
        return samples

    @staticmethod
    def get_files_in_directory(directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    @staticmethod
    def get_directories_in_directory(directory):
        return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    @staticmethod
    def get_latest_version():
        # outdated was for tensorboard
        # return sorted([int(f.split("_")[1]) for f in Utility.get_directories_in_directory("./lightning_logs")])[-1]

        # now wandb, return the entire file name
        return sorted(Utility.get_directories_in_directory("./wandb"))[-1] # eg: "run-20211007_123456-3k4j5l6m"

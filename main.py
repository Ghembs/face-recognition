from __future__ import print_function
import os
from identificator import Identificator
from utils import Configuration

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tensorflow warning removal


def main():
    config = Configuration()
    config.read_config()

    identity = Identificator(config.confidence, config.threshold, config.haar_path,
                             config.vgg_path, config.performance, video_path = config.video_path)
    identity.loop_frames()


if __name__ == "__main__":
    main()

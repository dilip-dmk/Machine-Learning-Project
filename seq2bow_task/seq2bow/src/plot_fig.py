import sys, os

from config import Configuration

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)
from plot import train_plot


if __name__ == "__main__":
	config = Configuration()
	train_plot(config)

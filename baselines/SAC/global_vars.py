from torch import cuda

SEED = 0
DT = 1/60

BATCH_SIZE = 128
NUM_EPISODES = 50
NUM_STEPS = 250

USE_CUDA = cuda.is_available()
USE_CUDA = False # BECAUSE OF MY PC's CUDA DEVICE BEING TOO OLD. REMOVE THIS LINE IF GPU IS UPDATED

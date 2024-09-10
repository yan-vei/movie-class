from dotenv import load_dotenv
import os


load_dotenv()

# Hyperparameters and other params from .env file and defaults
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
NUM_EPOCHS = int(os.getenv("EPOCHS", 5))
MAX_LENGTH = int(os.getenv("MAX_LEN", 512))
PADDING_TOKEN = str(os.getenv("PADDING_TOKEN", "-100"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 2))
POS_DIR_TRAIN = str(os.getenv("POS_DIR_TRAIN","aclImdb/train/pos"))
NEG_DIR_TRAIN = str(os.getenv("NEG_DIR_TRAIN", "aclImdb/train/neg"))
POS_DIR_TEST = str(os.getenv("POS_DIR_TEST", "aclImdb/test/pos"))
NEG_DIR_TEST = str(os.getenv("NEG_DIR_TEST", "aclImdb/test/neg"))
TOKENIZER = str(os.getenv("TOKENIZER", "bert-base-cased"))

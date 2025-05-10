import itertools
import copy
import os
import sys
import time
import shutil
import subprocess
import argparse
import numpy as np, random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Input, MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple, Any, Callable
import math
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.random.seed(42)
random.seed(42)
tf.keras.utils.set_random_seed(42)
for key, val in [
    ('font.family', 'sans-serif'),
    ('font.sans-serif', ['Arial', 'Liberation Sans', 'sans-serif'])
]:
    try:
        matplotlib.rcParams[key] = val
    except KeyError:
        print(f"[WARN] rcParam {key!r} not recognized, skipping")

print("\n=== TensorFlow GPU Check ===")
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
else:
    print("No GPUs detected — using CPU only!")
print("============================\n")

# =====================================================
# Hyperparameters & Global Weighting Settings
# =====================================================
DEBUG = True
TRIGGER_MODE = "auto"
TRIGGER_COLOR = "red"
BASE_EPOCHS = 1      
FT_EPOCHS = BASE_EPOCHS // 1        
N_CLIENTS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
FORGET_RATIO = 0.01
TRIGGER_LABEL = 9
TRIGGER_PATTERN = "cross"  
TRIGGER_PATCH_SIZE = 5
TRIGGER_VALUE = 1.0
DIRCHLET_ALPHA = 10000
SUBSET_RATIO = 0.01  
EDGE_BOOST = 1.5
WEIGHTING_SCHEME = "U"   
DEVALUATION = 0.25       

pca_cache = {}

# =====================================================
# Command-Line Argument Parsing and Validation
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Select unlearning strategies, hyperparameters, dataset, and weighting options to run. Fine-tuning is automatically executed."
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="NoOrth,ProjRaw,PCA,GS,PCA+GS",
        help="Comma-separated list of unlearning strategies to use. Options: NoOrth, ProjRaw, PCA, GS, PCA+GS"
    )
    parser.add_argument("--base_epochs", type=int, default=10, help="Number of global training epochs")
    parser.add_argument("--ft_epochs", type=int, default=None, help="Number of fine-tuning epochs (if not provided, computed as base_epochs//5)")
    parser.add_argument("--n_clients", type=int, default=5, help="Number of clients to distribute data among")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizers")
    parser.add_argument("--forget_ratio", type=float, default=None, help=...)
    parser.add_argument("--trigger_label", type=int, default=9, help="Label used when inserting backdoor triggers")
    parser.add_argument("--trigger_pattern", type=str, default="cross", help="Trigger pattern type ('cross' or 'corner')")
    parser.add_argument("--trigger_patch_size", type=int, default=5, help="Size of the trigger patch")
    parser.add_argument("--trigger_value", type=float, default=1.0, help="Pixel value for the trigger")
    parser.add_argument("--trigger_mode", type=str, choices=["auto","grayscale","color"], default="auto",
                        help="Backdoor trigger style: 'grayscale', 'color', or 'auto' detect by image channels")
    parser.add_argument("--trigger_color", type=str, choices=["red","blue","yellow","checker"], default="red",
                        help="Color of the trigger patch when using color mode")
    parser.add_argument("--dirchlet_alpha", type=float, default=10000, help="Alpha parameter for Dirichlet data partitioning")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Fraction of dataset used for training (if applicable)")

    # Allow multiple values for edge_boost, weighting_schemes, and devaluation:
    parser.add_argument("--edge_boost",
                        nargs='+',
                        type=float,
                        default=[4],
                        help="Boost factor(s) for edge weighting. Provide one or more values (e.g., --edge_boost 4 5)")
    parser.add_argument("--weighting_schemes",
                        nargs='+',
                        choices=["U", "decreasing", "increasing", "none"],
                        default=["U"],
                        help="List of weighting schemes to use. Options: U, decreasing, increasing, none.")
    parser.add_argument("--devaluation",
                        nargs='+',
                        type=float,
                        default=[0.25],
                        help="Devaluation factor(s) between 0 and 0.99 for points.")
    parser.add_argument("--use_pca", type=bool, default=True, help="Flag to indicate whether to use PCA")
    parser.add_argument("--dataset", type=str, choices=["mnist","cifar10","ham10000"], default="mnist",
                        help="Dataset to use: 'mnist', 'cifar10', or 'ham10000'")
    parser.add_argument("--iterated_unlearning", type=int, default=1,
                        help="Number of iterative unlearning updates to apply (default: 1)")

    parser.add_argument(
    "--forget_class_list",
    type=str,
    default=None,
    help="Comma‑separated list of class indices to forget entirely, e.g. '0,2,3'"
    )
    parser.add_argument(
        "--forget_class_count",
        type=int,
        default=None,
        help="Number of classes to forget at random (ignored if --forget_class_list is given)"
    )
    parser.add_argument(
    "--client_blocks",
    type=int,
    default=2,
    help="How many residual blocks the *client* keeps (must be > 1 and < 8)."
    )
    parser.add_argument(
    "--no_backdoor",           #   <-- NEW
    action="store_true",
    help="Disable backdoor poisoning and backdoor-attack evaluation "
         "(default: backdoor is enabled)"
    )
    parser.add_argument(
    "--epochs_per_round",
    type=int,
    default=1,
    help="How many epochs each client trains before the next client starts "
         "(default = 1 → exactly the old behaviour)."
    )
    parser.add_argument("--optimizer",
        choices=["sgd","adam", "adamw"], default="sgd",
        help="Which optimizer to use (sgd with momentum & weight-decay or adam).")
    parser.add_argument("--momentum",
        type=float, default=0.9,
        help="Momentum for SGD.")
    parser.add_argument("--weight_decay",
        type=float, default=5e-4,
        help="Weight decay (L2) for optimizer.")
    parser.add_argument("--augment",
        action="store_true",
        help="If set, use random crop + horizontal flip on CIFAR-10.")
        
    args = parser.parse_args()
    selected_strats = [s.strip() for s in args.strategies.split(',')]
    validate_args(args)
    return args, selected_strats


def validate_args(args):
    if args.base_epochs <= 0:
        sys.exit("Error: base_epochs must be a positive integer.")
    if args.ft_epochs is not None and args.ft_epochs < 0:
        sys.exit("Error: ft_epochs must be non-negative.")
    if args.n_clients <= 0:
        sys.exit("Error: n_clients must be a positive integer.")
    if args.batch_size <= 0:
        sys.exit("Error: batch_size must be a positive integer.")
    if args.learning_rate <= 0:
        sys.exit("Error: learning_rate must be positive.")
    if args.forget_ratio is not None and not (0 <= args.forget_ratio <= 1):
        sys.exit("Error: forget_ratio must be between 0 and 1.")
    if args.trigger_label < 0:
        sys.exit("Error: trigger_label must be non-negative.")
    if args.trigger_pattern not in ["cross", "corner"]:
        sys.exit("Error: trigger_pattern must be either 'cross' or 'corner'.")
    if args.trigger_patch_size <= 0:
        sys.exit("Error: trigger_patch_size must be positive.")
    if not (0 <= args.trigger_value <= 1):
        sys.exit("Error: trigger_value must be between 0 and 1.")
    # Validate trigger_mode
    if args.trigger_mode not in ["auto", "grayscale", "color"]:
        sys.exit("Error: trigger_mode must be 'auto', 'grayscale', or 'color'.")
    # Validate trigger_color
    if args.trigger_color not in ["red", "blue", "yellow", "checker"]:
        sys.exit("Error: trigger_color must be one of 'red', 'blue', 'yellow', or 'checker'.")
    if args.dirchlet_alpha <= 0:
        sys.exit("Error: dirchlet_alpha must be positive.")
    if not (0 < args.subset_ratio <= 1):
        sys.exit("Error: subset_ratio must be within (0, 1].")
    for eb in args.edge_boost:
        if eb <= 0:
            sys.exit("Error: each edge_boost must be positive.")
    for dv in args.devaluation:
        if not (0 <= dv <= 0.99):
            sys.exit("Error: each devaluation factor must be between 0 and 0.99.")
    if args.client_blocks < 1 or args.client_blocks >= 8:
        sys.exit("Error: --client_blocks must be an integer > 1 and < 8 (ResNet-18 has 8 blocks).")
    # --- new validation ---
    if args.forget_class_list is not None or args.forget_class_count is not None:
       if args.forget_ratio is not None and args.forget_ratio != 0:
            sys.exit("Specify EITHER --forget_ratio OR a class‑based forgetting flag, not both.")
       if args.forget_class_count is not None and args.forget_class_count <= 0:
            sys.exit("--forget_class_count must be positive.")
    if args.epochs_per_round <= 0:
        sys.exit("--epochs_per_round must be a positive integer.")
    if args.base_epochs % args.epochs_per_round != 0:
        print("[WARN] base_epochs is not an integer multiple of epochs_per_round – "
          "the last round will run fewer epochs.", file=sys.stderr)


# =====================================================
# Write Hyperparameters to a Text File (with description)
# =====================================================

def write_hyperparameters_file(
        *,
        hp_path: str,
        selected_strats,          # list[str]
        args,                     # the argparse.Namespace
        client_blocks: int,       # --client_blocks
        forgotten_classes=None    # list[int] or None
):
    """
    Dump every run-time hyperparameter (not the defaults!) to <hp_path>.
    """

    # Compute defaults that depend on other flags
    ft_epochs_val = args.ft_epochs if args.ft_epochs is not None else args.base_epochs // 5

    # ── assemble the text ───────────────────────────────────────────
    hp_text = f"""Hyperparameters (generated {time.strftime('%Y-%m-%d %H:%M:%S')}):
------------------------------------------------------------
General
  DEBUG                : {DEBUG}
  Dataset              : {args.dataset}
  N_CLIENTS            : {args.n_clients}
  CLIENT_BLOCKS        : {client_blocks}   # residual blocks on the client
  BASE_EPOCHS          : {args.base_epochs}
  FT_EPOCHS            : {ft_epochs_val}
  EPOCHS_PER_ROUND     : {args.epochs_per_round}
  BATCH_SIZE           : {args.batch_size}
  LEARNING_RATE        : {args.learning_rate}

Forgetting / Backdoor
  FORGET_RATIO         : {args.forget_ratio}
  FORGET_CLASS_LIST    : {args.forget_class_list}
  FORGET_CLASS_COUNT   : {args.forget_class_count}
  TRIGGER_LABEL        : {args.trigger_label}
  TRIGGER_PATTERN      : {args.trigger_pattern}
  TRIGGER_PATCH_SIZE   : {args.trigger_patch_size}
  TRIGGER_VALUE        : {args.trigger_value}
  TRIGGER_MODE         : {args.trigger_mode}
  TRIGGER_COLOR        : {args.trigger_color}
  NO_BACKDOOR          : {args.no_backdoor}

Data partitioning & subset
  DIRICHLET_ALPHA      : {args.dirchlet_alpha}
  SUBSET_RATIO         : {args.subset_ratio}

Weighting (unlearning)
  EDGE_BOOST values    : {args.edge_boost}
  Weighting Schemes    : {args.weighting_schemes}
  Devaluation Factors  : {args.devaluation}

PCA / Orthogonalisation
  USE_PCA              : {args.use_pca}

Iterative unlearning
  ITERATED_UNLEARNING  : {args.iterated_unlearning}

Selected unlearning strategies
  {', '.join(selected_strats)}

Forgotten classes (resolved)
  {forgotten_classes if forgotten_classes is not None else 'N/A'}
------------------------------------------------------------
"""

    # ── write to disk ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(hp_path), exist_ok=True)
    with open(hp_path, "w") as f:
        f.write(hp_text)
    print("Hyperparameters saved to:", hp_path, flush=True)




# =====================================================
# Data Loading and Preprocessing Functions
# =====================================================
def load_and_preprocess_mnist(subset_ratio=SUBSET_RATIO):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    if subset_ratio < 1.0:
        subset_size = int(len(X_train) * subset_ratio)
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    return X_train, y_train, X_test, y_test, X_test

def load_and_preprocess_cifar10(subset_ratio=1.0):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # normalize to zero-mean/unit-var per channel
    mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
    std  = np.std (X_train, axis=(0,1,2), keepdims=True)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test  = (X_test  - mean) / (std + 1e-7)

    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.125,   # 4/32
        height_shift_range=0.125,
        horizontal_flip=True,
        fill_mode='reflect'
    )
    datagen.fit(X_train)

    if subset_ratio < 1.0:
        subset_size = int(len(X_train) * subset_ratio)
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test,  num_classes=10)
    return X_train, y_train, X_test, y_test, X_test, datagen


def load_and_preprocess_ham10000(
    subset_ratio=1.0,
    image_size=(128, 128),
    batch_size=32,
    seed=123,
    ham_dir=None
):
    """
    Load HAM-10000 from a Kaggle dataset via CLI if not present locally.

    If `ham_dir` is not provided, defaults to ~/ham10000 and will be created.
    Expects inside `ham_dir`:
      - HAM10000_metadata.csv
      - image directories like HAM10000_images, HAM10000_images_part_1, etc.

    On first invocation, will download & unzip using Kaggle CLI.
    """
    # 1. Determine local directory, defaulting to ~/ham10000
    if ham_dir is None:
        ham_dir = os.path.expanduser("~/ham10000")
    # Create and/or download if missing or empty
    if not os.path.isdir(ham_dir) or not os.listdir(ham_dir):
        os.makedirs(ham_dir, exist_ok=True)
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "kmader/skin-cancer-mnist-ham10000",
            "--unzip", "-p", ham_dir
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download HAM10000 via Kaggle CLI: {e}")

    base_dir = ham_dir.rstrip('/')

    # 2. Load metadata
    meta_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
    if not os.path.isfile(meta_path):
        # maybe inside a subfolder
        for entry in os.listdir(base_dir):
            candidate = os.path.join(base_dir, entry, 'HAM10000_metadata.csv')
            if os.path.isfile(candidate):
                base_dir = os.path.dirname(candidate)
                meta_path = candidate
                break
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata CSV not found at {meta_path}")
    df = pd.read_csv(meta_path)

    # 3. Build file paths & labels
    # Detect image directories (any subdir containing 'image')
    img_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and 'image' in d.lower()]
    if not img_dirs:
        img_dirs = [base_dir]
    filepaths = []
    missing_ids = []
    for img_id in df['image_id']:
        filename = f"{img_id}.jpg"
        found = None
        for d in img_dirs:
            candidate = os.path.join(d, filename)
            if os.path.isfile(candidate):
                found = candidate
                break
        # fallback: remove 'ISIC_' prefix (if Kaggle names differ)
        if found is None and filename.startswith('ISIC_'):
            alt = filename.replace('ISIC_', '')
            for d in img_dirs:
                candidate = os.path.join(d, alt)
                if os.path.isfile(candidate):
                    found = candidate
                    break
        if found is None:
            missing_ids.append(img_id)
            filepaths.append(None)
        else:
            filepaths.append(found)
    if missing_ids:
        raise FileNotFoundError(
            f"Missing images for IDs: {missing_ids[:5]}... (total {len(missing_ids)})"
        )
    df['filepath'] = filepaths

    label_map = {cls: idx for idx, cls in enumerate(sorted(df['dx'].unique()))}
    df['label'] = df['dx'].map(label_map)

    # 4. Train/test split (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=seed
    )

    # 5. Optional subset of training set
    if subset_ratio < 1.0:
        train_df = train_df.sample(frac=subset_ratio, random_state=seed)

    # 6. Load images into NumPy arrays
    def load_images(df_subset):
        X, y = [], []
        for _, row in df_subset.iterrows():
            img = tf.io.read_file(row['filepath'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32) / 255.0
            X.append(img.numpy())
            y.append(row['label'])
        return np.stack(X), to_categorical(y, num_classes=len(label_map))

    X_train, y_train = load_images(train_df)
    X_test,  y_test  = load_images(test_df)

    # 7. Data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect'
    )
    datagen.fit(X_train)

    return X_train, y_train, X_test, y_test, datagen



# =====================================================
# Preprocessing Helper: Split Data Function
# =====================================================
def split_data(X, y, test_size=0.2, random_state=42):
    unique_ids = np.arange(len(X))
    return train_test_split(X, y, unique_ids, test_size=test_size, random_state=random_state)

# =====================================================
# Backdoor Trigger Functions (Updated for any image size)
# =====================================================
def insert_backdoor_trigger(images, labels,
                            trigger_label=TRIGGER_LABEL,
                            pattern=TRIGGER_PATTERN,
                            patch_size=TRIGGER_PATCH_SIZE,
                            trigger_value=TRIGGER_VALUE,
                            trigger_mode="auto",
                            trigger_color="red"):
    images = images.copy()
    labels = labels.copy()
    n = images.shape[0]

    def apply_color_patch(img, h_range, w_range, pattern, color, use_color):
        if img.ndim == 3 and img.shape[-1] == 3 and use_color:
            if pattern == "solid":
                if color == "red":
                    img[h_range[0]:h_range[1], w_range[0]:w_range[1], :] = [1.0, 0.0, 0.0]
                elif color == "blue":
                    img[h_range[0]:h_range[1], w_range[0]:w_range[1], :] = [0.0, 0.0, 1.0]
                elif color == "yellow":
                    img[h_range[0]:h_range[1], w_range[0]:w_range[1], :] = [1.0, 1.0, 0.0]
            elif pattern == "checker":
                for x in range(h_range[0], h_range[1]):
                    for y in range(w_range[0], w_range[1]):
                        img[x, y, :] = [1.0, 1.0, 0.0] if (x + y) % 2 == 0 else [0.0, 0.0, 1.0]
        else:
            # fallback to grayscale
            img[h_range[0]:h_range[1], w_range[0]:w_range[1]] = trigger_value
        return img

    # decide color vs grayscale
    if trigger_mode == "color":
        use_color = True
    elif trigger_mode == "grayscale":
        use_color = False
    else:  # auto detect
        use_color = (images.shape[-1] == 3)

    for i in range(n):
        h, w = images[i].shape[:2]
        if pattern == "corner":
            h0, h1 = h - patch_size, h
            w0, w1 = w - patch_size, w
            images[i] = apply_color_patch(images[i], (h0,h1), (w0,w1),
                                          "solid", trigger_color, use_color)
        elif pattern == "cross":
            ch, cw = h//2, w//2
            # horizontal bar
            images[i] = apply_color_patch(images[i], (ch, ch+patch_size), (0, w),
                                          "solid" if trigger_color!="checker" else "checker",
                                          trigger_color, use_color)
            # vertical bar
            images[i] = apply_color_patch(images[i], (0, h), (cw, cw+patch_size),
                                          "solid" if trigger_color!="checker" else "checker",
                                          trigger_color, use_color)

        # overwrite label
        if labels.ndim > 1:
            labels[i,:] = 0
            labels[i, trigger_label] = 1
        else:
            labels[i] = trigger_label

    return images, labels


def has_backdoor(image,
                 pattern,
                 patch_size,
                 trigger_mode="auto",
                 trigger_color="red",
                 trigger_value=1.0):
    """
    Detect whether `image` contains the backdoor trigger.
    Supports:
      - grayscale (scalar trigger_value)
      - solid-color (red/blue/yellow) and checkerboard patterns
    """
    h, w = image.shape[:2]

    # decide whether to check color or grayscale
    if trigger_mode == "color":
        use_color = True
    elif trigger_mode == "grayscale":
        use_color = False
    else:  # auto
        use_color = (image.ndim == 3 and image.shape[-1] == 3)

    # helper to compare a patch to expected RGB or grayscale
    def patch_matches(img_patch):
        if use_color and img_patch.ndim == 3:
            if trigger_color == "checker":
                # check alternating yellow/blue
                for x in range(img_patch.shape[0]):
                    for y in range(img_patch.shape[1]):
                        expected = np.array([1.0,1.0,0.0]) if ((x+y)%2==0) else np.array([0.0,0.0,1.0])
                        if not np.allclose(img_patch[x,y], expected):
                            return False
                return True
            else:
                color_map = {
                    "red":    np.array([1.0,0.0,0.0]),
                    "blue":   np.array([0.0,0.0,1.0]),
                    "yellow": np.array([1.0,1.0,0.0]),
                }
                expected = color_map[trigger_color]
                return np.allclose(img_patch, expected)
        else:
            # grayscale
            return np.allclose(img_patch, trigger_value)

    if pattern == "corner":
        patch = image[h-patch_size:h, w-patch_size:w, ...]
        return patch_matches(patch)

    elif pattern == "cross":
        # horizontal bar
        center_h = h // 2
        horiz = image[center_h:center_h+patch_size, :, ...]
        # vertical bar
        center_w = w // 2
        vert  = image[:, center_w:center_w+patch_size, ...]
        return patch_matches(horiz) or patch_matches(vert)

    else:
        return False


# =====================================================
# Unlearning Aggregation Strategies (Base Functions)
# =====================================================
@tf.function(jit_compile=True)
def aggregate_no_orth(stacked: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(stacked, axis=0)

@tf.function(jit_compile=True)
def aggregate_proj_raw(stacked: tf.Tensor) -> tf.Tensor:
    G = tf.reduce_sum(stacked, axis=0)

    def body(i, Gc):
        u = stacked[i]
        norm2 = tf.tensordot(u, u, axes=1)

        # named branch functions to avoid Autograph lambda bug
        def true_fn():
            return (tf.tensordot(Gc, u, axes=1) / norm2) * u

        def false_fn():
            return tf.zeros_like(u)

        proj = tf.cond(norm2 > 1e-10, true_fn, false_fn)
        return i + 1, Gc - proj

    _, G_final = tf.while_loop(
        lambda i, _: i < tf.shape(stacked)[0],
        body,
        (0, G),
        parallel_iterations=1
    )
    return G_final


@tf.function(jit_compile=True)
def aggregate_gs_no_proj(stacked: tf.Tensor) -> tf.Tensor:
    """
    GS‑no‑proj in pure TF:  
    • flatten each gradient into a vector  
    • do QR on the transpose to get an orthonormal basis of the row‑space  
    • sum those basis vectors  
    • reshape back to the original variable shape
    """
    # stacked: shape = [n_grads, ...var_shape...]
    s = tf.shape(stacked)
    n = s[0]
    # flatten each grad into length‑D vector
    flat = tf.reshape(stacked, [n, -1])         # [n, D]
    # we want an orthonormal basis of the row‑space of `flat`
    # so do QR on flatᵀ: shape [D, n], Q columns are that basis
    flat_T = tf.transpose(flat)                 # [D, n]
    Q, _ = tf.linalg.qr(flat_T, full_matrices=False)  # Q: [D, n]
    # sum the n basis vectors (the columns of Q)
    basis_sum = tf.reduce_sum(Q, axis=1)        # [D]
    # reshape back into the original var shape
    return tf.reshape(basis_sum, s[1:])        # [...var_shape...]

@tf.function(jit_compile=True)
def aggregate_identity(stacked: tf.Tensor) -> tf.Tensor:
    # just sum up nothing (i.e. do no unlearning)
    # we return zero update so the weights never change
    return tf.zeros_like(tf.reduce_sum(stacked, axis=0))


@tf.function(jit_compile=True)
def aggregate_pca(stacked: tf.Tensor) -> tf.Tensor:
    G = tf.reduce_sum(stacked, axis=0)
    # SVD: stacked = U Σ Vᵀ
    s, u, v = tf.linalg.svd(stacked, full_matrices=False)
    # pick the first right‐singular vector (length = feature_dim)
    pc1 = v[:, 0]
    proj = tf.tensordot(G, pc1, axes=1) * pc1
    return G - proj


@tf.function(jit_compile=True)
def aggregate_pca_gs(stacked: tf.Tensor) -> tf.Tensor:
    G = tf.reduce_sum(stacked, axis=0)
    # take up to two right‐singular vectors
    s, u, v = tf.linalg.svd(stacked, full_matrices=False)
    # v has shape [feature_dim, k], so columns are PCs
    k = tf.shape(v)[1]
    take = tf.minimum(2, k)
    pcs = v[:, :take]   # each column is a principal component
    for i in range(take):
        pc = pcs[:, i]
        G = G - (tf.tensordot(G, pc, axes=1) * pc)
    return G


def weighted_aggregate(stacked, strategy_func, scheme=WEIGHTING_SCHEME, edge_boost=EDGE_BOOST, devaluation=DEVALUATION):
    n = stacked.shape[0]
    weights = tf.convert_to_tensor(compute_weights(n, scheme, edge_boost, devaluation), dtype=stacked.dtype)
    weights = tf.expand_dims(weights, axis=1)
    weighted_stacked = stacked * weights
    return strategy_func(weighted_stacked)

# =====================================================
# Weighting Computation Function
# =====================================================
def compute_weights(n, scheme, edge_boost, devaluation):
    """
    Computes weights for n batches according to the selected scheme.

    - U: U-shaped weighting. Edges get weight=1 and the middle gets devaluation.
    - decreasing: Linear decrease from 1 (first batch) to devaluation (last batch).
    - increasing: Linear increase from devaluation (first batch) to 1 (last batch).
    - none: No weighting (all weights are 1).

    All computed weights are then multiplied by the edge_boost factor.
    """
    if n <= 1:
        return np.array([edge_boost], dtype=np.float32)
    if scheme == "U":
        mid = (n - 1) / 2.0
        weights = np.array([devaluation + (1 - devaluation) * (abs(i - mid) / mid) for i in range(n)])
    elif scheme == "decreasing":
        weights = np.array([1 - (1 - devaluation) * (i / (n - 1)) for i in range(n)])
    elif scheme == "increasing":
        weights = np.array([devaluation + (1 - devaluation) * (i / (n - 1)) for i in range(n)])
    elif scheme == "none":
        weights = np.ones(n, dtype=np.float32)
    else:
        raise ValueError("Unknown weighting scheme.")
    weights = weights * edge_boost
    return weights.astype(np.float32)



# =====================================================
# Modified Unlearning Functions with Weighting
# =====================================================
@tf.function
def aggregate_gs(stacked):
    """
    Aggregates gradients using orthogonal projection with XLA-safe TensorFlow ops.
    Compatible with TF 2.11.1 and optimized for GPU.
    """
    G = tf.reduce_sum(stacked, axis=0)
    n = tf.shape(stacked)[0]

    ta = tf.TensorArray(dtype=stacked.dtype, size=n, dynamic_size=False, clear_after_read=False)
    size = tf.constant(0)

    def cond(i, ta, size):
        return i < n

    def body(i, ta, size):
        v = stacked[i]

        def project(j, v):
            u = ta.read(j)
            dot = tf.reduce_sum(v * u)
            v = v - dot * u
            return j + 1, v

        j = tf.constant(0)
        def inner_cond(j, _): return j < size
        j, v = tf.while_loop(inner_cond, project, [j, v],
                             shape_invariants=[j.get_shape(), tf.TensorShape(None)])

        norm = tf.norm(v)
        should_add = norm > 1e-10

        def add_vec():
            u = v / norm
            return ta.write(size, u), size + 1

        def skip_vec():
            return ta, size

        ta, size = tf.cond(should_add, add_vec, skip_vec)
        return i + 1, ta, size

    i0 = tf.constant(0)
    i_final, ta_final, size_final = tf.while_loop(
        cond,
        body,
        [i0, ta, size],
        shape_invariants=[i0.get_shape(), tf.TensorShape(None), size.get_shape()]
    )

    def use_first_u():
        u0 = ta_final.read(0)
        dot = tf.reduce_sum(G * u0)
        return G - dot * u0

    def use_zero():
        return tf.zeros_like(G)

    return tf.cond(size_final > 0, use_first_u, use_zero)



def safe_assign_add(var, update, lr=1.0, max_norm=5.0):
    norm = tf.norm(update)
    if tf.math.is_finite(norm) and norm > max_norm:
        update = update * (max_norm / (norm + 1e-12))
    if tf.reduce_any(tf.math.is_nan(update)):
        tf.print("[WARNING] NaNs in update, skipping")
        return
    var.assign_add(lr * update)

def weighted_aggregate(stacked, strategy_func, scheme=WEIGHTING_SCHEME, edge_boost=EDGE_BOOST, devaluation=DEVALUATION):
    n = stacked.shape[0]
    weights = tf.convert_to_tensor(compute_weights(n, scheme, edge_boost, devaluation), dtype=stacked.dtype)
    weights = tf.expand_dims(weights, axis=1)
    weighted_stacked = stacked * weights
    return strategy_func(weighted_stacked)

def unlearn_server_with_strategy(
        server_model, server_cache, forget_batches, strategy_func,
        learning_rate=LEARNING_RATE, *,
        scheme, edge_boost, devaluation, chunk_size=200):
    trainable_vars   = server_model.trainable_variables
    accumulated_upd  = [tf.zeros_like(w) for w in trainable_vars]
    total_batches = len(forget_batches)
    total_chunks = math.ceil(total_batches / chunk_size)
    print(f"[DEBUG] Server unlearning: {total_batches} batches to forget → {total_chunks} chunks (chunk_size={chunk_size})", flush=True)
    
    # ── stream over shards ──────────────────────────────────────────────
    for grad_chunk in stream_grad_chunks(sorted(forget_batches), server_cache, chunk_size):
        # per-variable aggregation on THIS chunk
        for w_idx in range(len(accumulated_upd)):
            vecs = [tf.reshape(g[w_idx], [-1])
                    for g in grad_chunk
                    if g[w_idx].shape == accumulated_upd[w_idx].shape]
            if not vecs:
                continue
            stacked = tf.stack(vecs, axis=0)         # ≤ chunk_size rows
            upd     = weighted_aggregate(
                          stacked, strategy_func,
                          scheme=scheme,
                          edge_boost=edge_boost,
                          devaluation=devaluation)
            upd = tf.reshape(upd, accumulated_upd[w_idx].shape)
            accumulated_upd[w_idx] += upd             # running total
    # ── apply to weights ────────────────────────────────────────────────
    for var, upd in zip(trainable_vars, accumulated_upd):
        safe_assign_add(var, upd, learning_rate)
    
    return server_model


def unlearn_client_with_strategy(
        client_model, client_cache, client_id, forget_batches, strategy_func,
        learning_rate=LEARNING_RATE, *, logging,
        scheme, edge_boost, devaluation, chunk_size=200):

    trainable_vars   = client_model.trainable_variables
    accumulated_upd  = [tf.zeros_like(w) for w in trainable_vars]
    total_batches = len(forget_batches)
    total_chunks = math.ceil(total_batches / chunk_size)
    print(f"[DEBUG] Client {client_id} unlearning: {total_batches} batches → {total_chunks} chunks", flush=True)
 

    # All cached shard names, sorted once
    cache_keys = sorted(client_cache.keys())
    # Map a target batch → the shard we'll actually use
    chosen = [find_closest_batch(cache_keys, t) for t in sorted(forget_batches)]
    chosen = [c for c in chosen if c is not None]          # drop misses

    for grad_chunk in stream_grad_chunks(chosen, client_cache, chunk_size):
        for w_idx in range(len(accumulated_upd)):
            vecs = [tf.reshape(g[w_idx], [-1])
                    for g in grad_chunk
                    if g[w_idx].shape == accumulated_upd[w_idx].shape]
            if not vecs:
                continue
            stacked = tf.stack(vecs, axis=0)
            upd     = weighted_aggregate(
                          stacked, strategy_func,
                          scheme=scheme,
                          edge_boost=edge_boost,
                          devaluation=devaluation)
            upd = tf.reshape(upd, accumulated_upd[w_idx].shape)
            accumulated_upd[w_idx] += upd

    for var, upd in zip(trainable_vars, accumulated_upd):
        safe_assign_add(var, upd, learning_rate)

    return client_model


def find_closest_batch(existing_batches, target_batch):
    sorted_batches = sorted(existing_batches)
    for b in sorted_batches:
        if b >= target_batch:
            return b
    return None

# =====================================================
# Model Building for Split Learning
# =====================================================
def basic_block(x, filters, stride=1):
    shortcut = x
    # first conv
    x = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # second conv
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    # adjust shortcut if needed
    if stride!=1 or shortcut.shape[-1]!=filters:
        shortcut = Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    return Activation('relu')(x)
# helper lists describe the 8 residual blocks in ResNet-18
_BLOCK_FILTERS = [64, 64, 128, 128, 256, 256, 512, 512]
_BLOCK_STRIDES = [1 ,  1 ,  2  , 1  , 2  , 1  , 2  , 1  ]  # stride on *first* conv

def create_client_model(input_shape, num_client_blocks=2):
    """Stem + the first `num_client_blocks` residual blocks."""
    inputs = Input(shape=input_shape)

    # stem
    x = Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(3, strides=2, padding="same")(x)

    # prefix of ResNet-18
    for i in range(num_client_blocks):
        x = basic_block(x,
                        filters=_BLOCK_FILTERS[i],
                        stride=_BLOCK_STRIDES[i])

    return Model(inputs, x)


def create_server_model(
    client_output_shape,
    num_classes,
    num_client_blocks: int = 2,
    optimizer_fn: Callable[[float], tf.keras.optimizers.Optimizer] = None,
    learning_rate: float = 1e-3
):
    """The remaining (8 − num_client_blocks) residual blocks + head."""
    inputs = Input(shape=client_output_shape)
    x = inputs

    for i in range(num_client_blocks, 8):
        x = basic_block(x,
                        filters=_BLOCK_FILTERS[i],
                        stride=_BLOCK_STRIDES[i])

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    # compile with the optimizer_fn you pass in
    model.compile(
        optimizer=optimizer_fn(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model

def cutout_batch(X, mask_size=8):
    # X: numpy array, shape (B,H,W,C)
    B, H, W, _ = X.shape
    for i in range(B):
        y = np.random.randint(H)
        x = np.random.randint(W)
        y1, y2 = np.clip([y-mask_size//2, y+mask_size//2], 0, H)
        x1, x2 = np.clip([x-mask_size//2, x+mask_size//2], 0, W)
        X[i, y1:y2, x1:x2, :] = 0
    return X

def mixup_batch(X, Y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X))
    X_mix = lam * X + (1-lam) * X[idx]
    Y_mix = lam * Y + (1-lam) * Y[idx]
    return X_mix, Y_mix

def stream_grad_chunks(forget_batches, cache, chunk_size=200):
    chunk = []
    for b in forget_batches:
        if b in cache:
            chunk.append(_load_grad_list(cache[b]))
            del cache[b]
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:                      
        yield chunk

# =====================================================
# Optimized Training: Sequential Split Learning (Clients train one-by-one)
# =====================================================
def train_sequential(
        clients: List[tf.keras.Model],
        server_model: tf.keras.Model,
        client_data: List[np.ndarray],
        client_labels: List[np.ndarray],
        client_ids: List[np.ndarray],
        *,                             # ---------------- optional args
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
        X_bd:    np.ndarray | None = None,
        y_bd:    np.ndarray | None = None,
        forget_data:   List[np.ndarray] | None = None,    # NEW
        forget_labels: List[np.ndarray] | None = None,    # NEW
        retain_data:   List[np.ndarray] | None = None,    # NEW
        retain_labels: List[np.ndarray] | None = None,    # NEW
        batch_size: int = 32,
        base_epochs: int = 1,
        epochs_per_round: int = 1,
        cache_gradients: bool = False,
        log_dict: Dict | None = None,
        learning_rate: float = 0.001,
        optimizer_fn: Callable[[float], tf.keras.optimizers.Optimizer] = None,
        datagen: Optional[ImageDataGenerator] = None,
        DEBUG: bool = True,
        GRAD_DIR: str = "grad_shards"
    ) -> Tuple[Dict[int, List[int]], Dict[int, str],
               List[Dict[int, str]], List[List[Dict[str, Any]]]]:
    """
    Split‑learning training loop **with per‑epoch metrics** for
      – validation set (`val_acc`)
      – each client's FORGET set (`forget_acc`)
      – each client's RETAIN set (`retain_acc`)
    The new metrics are stored in `hist` and printed live.
    """

    assert epochs_per_round > 0 and base_epochs > 0, "epochs must be positive"
    n_clients = len(clients)
    n_rounds  = int(math.ceil(base_epochs / epochs_per_round))

    # 1. build optimisers ---------------------------------------------------
    client_opts = [optimizer_fn(learning_rate) for _ in clients]
    server_opt  = optimizer_fn(learning_rate)
    for opt, cm in zip(client_opts, clients):
        opt.build(cm.trainable_variables)
    server_opt.build(server_model.trainable_variables)

    # 2. bookkeeping for unlearning / caching ------------------------------
    server_cache: Dict[int, str]      = {}
    client_caches: List[Dict[int,str]]= [dict() for _ in clients]
    batch_map: Dict[int, List[int]]   = {}
    batch_ctr = tf.Variable(0, tf.int32)

    # 3. per‑client epoch counters & history -------------------------------
    done_epochs = [0] * n_clients
    hist: List[List[Dict[str, Any]]] = [[] for _ in clients]
    
    # ── record epoch 0 metrics ────────────────────────────────────────────
    for cid, cm in enumerate(clients):
        rec = {"epoch": 0, "loss": float("nan")}
        # validation
        if X_val is not None and y_val is not None:
            pv  = cm.predict(X_val, verbose=0)
            sv  = server_model.predict(pv, verbose=0)
            rec["val_acc"] = np.mean(np.argmax(sv,1) == np.argmax(y_val,1))
        else:
            rec["val_acc"] = float("nan")
        # FORGET set
        if forget_data is not None and len(forget_data[cid]) > 0:
            pf  = cm.predict(forget_data[cid], verbose=0)
            sf  = server_model.predict(pf, verbose=0)
            rec["forget_acc"] = accuracy_from_preds(sf, forget_labels[cid])
        else:
            rec["forget_acc"] = float("nan")
        # RETAIN set
        if retain_data is not None and len(retain_data[cid]) > 0:
            pr  = cm.predict(retain_data[cid], verbose=0)
            sr  = server_model.predict(pr, verbose=0)
            rec["retain_acc"] = accuracy_from_preds(sr, retain_labels[cid])
        else:
            rec["retain_acc"] = float("nan")

        hist[cid].append(rec)
        if log_dict is not None:
            log_dict.setdefault(cid, []).append(rec)
        print(f"[client {cid}] epoch {rec['epoch']:3d}  "
              f"val={rec['val_acc']:.4f}  "
              f"forget={rec['forget_acc']:.4f}  "
              f"retain={rec['retain_acc']:.4f}",
              flush=True)
    # 4. main loop ----------------------------------------------------------
    for rnd in range(1, n_rounds + 1):
        client_order = np.random.permutation(n_clients)
        if DEBUG:
            print(f"\n[Round {rnd}/{n_rounds}] client order → {client_order}", flush=True)

        for cid in client_order:
            need = min(epochs_per_round, base_epochs - done_epochs[cid])
            if need == 0:
                continue

            cm         = clients[cid]
            Xc, yc, idc= client_data[cid], client_labels[cid], client_ids[cid]
            
            if DEBUG:
                print(f"→ client {cid}: training {need} epoch(s) "
                      f"(done {done_epochs[cid]}/{base_epochs})", flush=True)

            for _ in range(need):
                done_epochs[cid] += 1

                # --- (mini)batch iterator ---------------------------------
                perm        = np.random.permutation(len(Xc))
                Xc_shuf     = Xc[perm]
                yc_shuf     = yc[perm]
                idc_shuf    = idc[perm]

                if datagen:
                    n_steps = math.ceil(len(Xc_shuf) / batch_size)
                    flow    = datagen.flow(Xc_shuf, yc_shuf,
                                           batch_size=batch_size,
                                           shuffle=False)
                    batch_iter = (
                        (xb, yb, idc_shuf[i*batch_size:(i+1)*batch_size])
                        for i in range(n_steps)
                        for xb, yb in [flow.next()]
                    )
                else:
                    def batch_iter():
                        for i in range(0, len(Xc_shuf), batch_size):
                            yield (
                                Xc_shuf[i:i+batch_size],
                                yc_shuf[i:i+batch_size],
                                idc_shuf[i:i+batch_size],
                            )
                    batch_iter = batch_iter()

                # --- minibatch loop ---------------------------------------
                epoch_loss = 0.0
                steps      = 0

                for Xb_np, yb_np, idb in batch_iter:
                    if datagen is not None:
                        # Cutout: zero-out one 8×8 patch per image
                        Xb_np = cutout_batch(Xb_np, mask_size=8)
                
                        # Mixup: mix each image+label with another at α=0.2
                        Xb_np, yb_np = mixup_batch(Xb_np, yb_np, alpha=0.2)
                        
                    Xb = tf.convert_to_tensor(Xb_np)
                    yb = tf.convert_to_tensor(yb_np)

                    # record batch id per sample for unlearning
                    bidx = int(batch_ctr.numpy())
                    for uid in idb:
                        batch_map.setdefault(uid, []).append(bidx)

                    loss, g_srv, g_cli = train_step(
                        cm, server_model, Xb, yb,
                        client_opts[cid], server_opt
                    )

                    # optional gradient caching
                    if cache_gradients:
                        os.makedirs(GRAD_DIR, exist_ok=True)
                        srv_shard = os.path.join(GRAD_DIR, f"srv_{bidx}.npz")
                        cli_shard = os.path.join(GRAD_DIR, f"cli{cid}_{bidx}.npz")
                        np.savez_compressed(srv_shard, *[g.numpy() for g in g_srv])
                        np.savez_compressed(cli_shard, *[g.numpy() for g in g_cli])
                        server_cache[bidx]       = srv_shard
                        client_caches[cid][bidx] = cli_shard

                    batch_ctr.assign_add(1)
                    epoch_loss += float(loss)
                    steps      += 1

                # ---------- epoch‑end metrics -----------------------------
                rec = {
                    "epoch": done_epochs[cid],
                    "loss":  epoch_loss / max(1, steps)
                }

                # 1) validation / test set
                if X_val is not None and y_val is not None:
                    pv   = cm.predict(X_val, verbose=0)
                    sv   = server_model.predict(pv, verbose=0)
                    probs= tf.nn.softmax(sv, axis=1).numpy()
                    rec["val_acc"] = np.mean(
                        np.argmax(probs,1) == np.argmax(y_val,1)
                    )
                else:
                    rec["val_acc"] = float('nan')

                # 2) FORGET set
                if forget_data is not None and len(forget_data[cid]) > 0:
                    pf   = cm.predict(forget_data[cid], verbose=0)
                    sf   = server_model.predict(pf, verbose=0)
                    rec["forget_acc"] = accuracy_from_preds(sf, forget_labels[cid])
                else:
                    rec["forget_acc"] = float('nan')

                # 3) RETAIN set
                if retain_data is not None and len(retain_data[cid]) > 0:
                    pr   = cm.predict(retain_data[cid], verbose=0)
                    sr   = server_model.predict(pr, verbose=0)
                    rec["retain_acc"] = accuracy_from_preds(sr, retain_labels[cid])
                else:
                    rec["retain_acc"] = float('nan')

                # store + optional live print
                hist[cid].append(rec)
                if log_dict is not None:
                    log_dict.setdefault(cid, []).append(rec)

                print(f"[client {cid}] epoch {rec['epoch']:3d}  "
                      f"val={rec['val_acc']:.4f}  "
                      f"forget={rec['forget_acc']:.4f}  "
                      f"retain={rec['retain_acc']:.4f}",
                      flush=True)

        if all(d == base_epochs for d in done_epochs):
            break

    return batch_map, server_cache, client_caches, hist





@tf.function(jit_compile=True, reduce_retracing=True)
def train_step(client_model, server_model, X_batch, y_batch, client_optimizer, server_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        client_out = client_model(X_batch, training=True)
        server_out = server_model(client_out, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, server_out))
    grads_server = tape.gradient(loss, server_model.trainable_variables)
    grads_client = tape.gradient(loss, client_model.trainable_variables)
    grads_server = [tf.clip_by_norm(g, 1.0) for g in grads_server]
    grads_client = [tf.clip_by_norm(g, 1.0) for g in grads_client]
    server_optimizer.apply_gradients(zip(grads_server, server_model.trainable_variables))
    client_optimizer.apply_gradients(zip(grads_client, client_model.trainable_variables))
    return loss, grads_server, grads_client
# Save gradients somewhere else
def _save_grad_list(grads, shard_path, fp16=True):
    """Store list[Tensor] → disk; returns shard_path."""
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    arrs = [g.numpy().astype(np.float16 if fp16 else np.float32) for g in grads]
    np.savez_compressed(shard_path, *arrs)
    return shard_path

def _load_grad_list(shard_path):
    """Load list[np.ndarray] from disk → list[tf.Tensor], cast to float32."""
    data = np.load(shard_path)
    grads = []
    for key in data.files:
        arr_fp16 = data[key]
        arr_fp32 = arr_fp16.astype(np.float32)        # ← cast here
        grads.append(tf.convert_to_tensor(arr_fp32))  # now dtype=float32
    return grads

# ----------------- Helper Functions for Cached Evaluations -----------------
def accuracy_from_preds(server_preds, onehot_labels):
    probs = tf.nn.softmax(server_preds, axis=1).numpy()
    preds = np.argmax(probs, axis=1)
    true_labels = np.argmax(onehot_labels, axis=1)
    return np.mean(preds == true_labels)

def mia_auc(scores_member, scores_nonmember):
    all_scores = np.concatenate([scores_member, scores_nonmember])
    all_labels = np.concatenate([np.ones_like(scores_member), np.zeros_like(scores_nonmember)])
    
    finite = np.isfinite(all_scores)
    if np.sum(finite) < 2:
        return float("nan")  # Not enough valid points

    return roc_auc_score(all_labels[finite], all_scores[finite])


def evaluate_backdoor_attack_single_client(server_model, client_model,
                                           X_clean_test, *,
                                           enable_backdoor: bool,
                                           pattern=TRIGGER_PATTERN,
                                           trigger_label=TRIGGER_LABEL):
    if not enable_backdoor:
        return 0.0

    # ─── Generate backdoored test set exactly as at training time ─────────
    X_bd, _ = insert_backdoor_trigger(
        X_clean_test.copy(),
        np.zeros((X_clean_test.shape[0],), dtype=int),   # dummy labels
        trigger_label  = trigger_label,
        pattern        = pattern,
        patch_size     = TRIGGER_PATCH_SIZE,
        trigger_value  = TRIGGER_VALUE,
        trigger_mode   = TRIGGER_MODE,
        trigger_color  = TRIGGER_COLOR
    )

    # ─── Now do your two-stage predict as before ─────────────────────────
    pred_bd       = client_model.predict(X_bd, verbose=0)
    server_preds  = server_model.predict(pred_bd, verbose=0)
    probs_bd      = tf.nn.softmax(server_preds, axis=1).numpy()
    return np.mean(np.argmax(probs_bd, axis=1) == trigger_label)

# =====================================================
# Improved MIA
# =====================================================

def mia_scores_all(member_scores: np.ndarray,
                   nonmember_scores: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Compute MIA AUC, accuracy, precision and recall for a *single* comparison.

    Returns
    -------
    dict  with keys  {"auc", "acc", "prec", "rec"}
    """
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones_like(member_scores),
        np.zeros_like(nonmember_scores)
    ])

    # keep only finite numbers
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]

    if scores.size < 2:  # degenerate case
        return {"auc": np.nan, "acc": np.nan, "prec": np.nan, "rec": np.nan}

    auc  = roc_auc_score(labels, scores)
    pred = (scores >= threshold).astype(int)

    acc  = accuracy_score (labels, pred)
    prec = precision_score(labels, pred, zero_division=0)
    rec  = recall_score   (labels, pred, zero_division=0)

    return {"auc": auc, "acc": acc, "prec": prec, "rec": rec}
# =====================================================
# Optimized Evaluation Function: Gather All Metrics (Cached Predicts)
# =====================================================
# ---------------------------------------------------------------------------
# 1)  Batch‑JIT TF feature extractor for (conf, entropy, margin)
# ---------------------------------------------------------------------------

@tf.function
def extract_attack_features_tf(client_model, server_model, X):
    """Return a (N,3) tf.Tensor with [confidence, entropy, margin] per sample."""
    mid = client_model(X, training=False)
    logits = server_model(mid, training=False)
    probs = tf.nn.softmax(logits, axis=1)

    conf = tf.reduce_max(probs, axis=1)
    ent = -tf.reduce_sum(probs * tf.math.log(probs + 1e-12), axis=1)
    top2 = tf.nn.top_k(probs, k=2).values  # shape=(batch,2)
    margin = top2[:,0] - top2[:,1]

    return tf.stack([conf, ent, margin], axis=1)

# ---------------------------------------------------------------------------
# 2)  TF‑native attack head training
# ---------------------------------------------------------------------------

def train_attack_head_tf(feats, labels, epochs=10, batch_size=64):
    """
    Train a simple 1-layer TF attack head on extracted features.
    feats: (N,3) np.ndarray or tf.Tensor, labels: binary {0,1}
    Returns trained Keras Model.
    """
    # ensure tf.Tensor
    feats = tf.convert_to_tensor(feats, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((feats, labels)) 
    dataset = dataset.shuffle(1024).batch(batch_size)

    attack = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    attack.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    attack.fit(dataset, epochs=epochs, verbose=0)
    return attack

# ---------------------------------------------------------------------------
# 3)  Optimized shadow‑model attack using TF pipeline
# ---------------------------------------------------------------------------

def shadow_attack_tf(client_model, server_model,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_target: np.ndarray, *, K: int = 5,
                     epochs_ft: int = 1) -> float:
    """
    Return AUC of a learned attack on X_target non‑members.

    1) Split X_train into K folds, for each fold treat it as 'members' and
       sample an equal‑sized non‑member set.
    2) Extract (conf, entropy, margin) features via extract_attack_features_tf.
    3) Train a 1‑layer attack head on those features.
    4) Evaluate on held‑out 25% of those features for val‑AUC.
    5) Finally, extract features on X_target and report AUC against label=0.
    """
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_train))
    splits = np.array_split(indices, K)

    feat_buffer = []
    lab_buffer  = []

    # 1. Build member/non‑member feature sets
    for mem_idx in splits:
        non_idx = rng.choice(
            np.setdiff1d(indices, mem_idx),
            size=len(mem_idx),
            replace=False
        )

        # Extract attack features for members and non‑members
        X_mem = tf.convert_to_tensor(X_train[mem_idx], tf.float32)
        X_non = tf.convert_to_tensor(X_train[non_idx], tf.float32)

        feat_mem = extract_attack_features_tf(client_model, server_model, X_mem).numpy()
        feat_non = extract_attack_features_tf(client_model, server_model, X_non).numpy()

        feat_buffer.append(feat_mem)
        feat_buffer.append(feat_non)
        lab_buffer.append(np.ones(len(feat_mem),  dtype=int))
        lab_buffer.append(np.zeros(len(feat_non), dtype=int))

    # 2. Stack into one dataset
    X_att = np.vstack(feat_buffer)
    y_att = np.concatenate(lab_buffer)

    # 3. Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_att, y_att,
        test_size=0.25,
        random_state=42,
        stratify=y_att
    )

    # 4. Train the 1‑layer attack head
    attack_model = train_attack_head_tf(X_tr, y_tr, epochs=epochs_ft, batch_size=32)
    val_loss, val_auc = attack_model.evaluate(
        tf.convert_to_tensor(X_val, tf.float32),
        y_val,
        verbose=0
    )
    tf.print(f'[Shadow‑MIA] Validation AUC:', val_auc)

    # 5. Final evaluation on X_target (all treated as "non‑member" class 0)
    Xt = tf.convert_to_tensor(X_target, tf.float32)
    Xt_feat = extract_attack_features_tf(client_model, server_model, Xt).numpy()
    probs   = attack_model.predict(Xt_feat, verbose=0).ravel()
    auc      = roc_auc_score(np.zeros_like(probs), probs)

    return auc

def compute_shadow_mia_metrics(
    client_model, server_model,
    X_members,           
    y_members,          
    X_pos_target,        
    X_neg_target,        
    K: int = 16,
    epochs_ft: int = 10,
    batch_size: int = 32
) -> dict:
    """
    Run a K-fold shadow attack and return averaged
    {'auc', 'acc', 'prec', 'rec'} on (X_pos_target vs X_neg_target).
    """
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_members))
    folds = np.array_split(idx, K)
    metrics = {'auc': [], 'acc': [], 'prec': [], 'rec': []}

    for fold in folds:
        # build member/non-member train split for this shadow
        non_idx = rng.choice(
            np.setdiff1d(idx, fold),
            size=len(fold),
            replace=False
        )
        # extract features (conf, ent, margin)
        feats_mem = extract_attack_features_tf(
            client_model, server_model,
            tf.convert_to_tensor(X_members[fold], tf.float32)
        ).numpy()
        feats_non = extract_attack_features_tf(
            client_model, server_model,
            tf.convert_to_tensor(X_members[non_idx], tf.float32)
        ).numpy()

        # train attack head
        X_att = np.vstack([feats_mem, feats_non])
        y_att = np.concatenate([np.ones(len(feats_mem)), np.zeros(len(feats_non))])
        attack = train_attack_head_tf(X_att, y_att, epochs=epochs_ft, batch_size=batch_size)

        # evaluate on target sets
        feats_pos = extract_attack_features_tf(
            client_model, server_model,
            tf.convert_to_tensor(X_pos_target, tf.float32)
        ).numpy()
        feats_neg = extract_attack_features_tf(
            client_model, server_model,
            tf.convert_to_tensor(X_neg_target, tf.float32)
        ).numpy()

        scores = np.concatenate([attack.predict(feats_pos).ravel(),
                                 attack.predict(feats_neg).ravel()])
        labels = np.concatenate([np.ones(len(feats_pos)), np.zeros(len(feats_neg))])
        preds  = (scores >= 0.5).astype(int)

        # collect fold metrics
        metrics['auc'].append(roc_auc_score(labels, scores))
        metrics['acc'].append(accuracy_score(labels, preds))
        metrics['prec'].append(precision_score(labels, preds, zero_division=0))
        metrics['rec'].append(recall_score(labels, preds, zero_division=0))

        # clear TF state so we don't leak VRAM
        tf.keras.backend.clear_session()

    # average across folds
    return {k: float(np.mean(v)) for k, v in metrics.items()}
    
# ---------------------------------------------------------------------------
# 4)  PATCHED gather_all_metrics  – include TF‑optimized shadow attack
# ---------------------------------------------------------------------------

def gather_all_metrics(server_model, clients,
                       client_data, client_labels,
                       X_forget, y_forget, id_forget, client_ids,
                       X_test,   y_test,X_test_retain, y_test_retain,
                       X_train, y_train, *,
                       enable_backdoor: bool = True):
    """
    Returns per-client metrics and prints a global summary including:
      - train/forget/test accuracy
      - simple MIA (AUC, acc, prec, rec) on train vs test & forget vs test
      - shadow-attack MIA (AUC, acc, prec, rec) on forget vs test
      - backdoor success rate
    """
    results = []

    for cid, client_model in enumerate(clients):
        # 1) Base accuracies
        pred_train = client_model.predict(client_data[cid], verbose=0)
        srv_train = tf.nn.softmax(server_model.predict(pred_train), axis=1).numpy()
        train_acc = accuracy_from_preds(srv_train, client_labels[cid])
        conf_train = np.max(srv_train, axis=1)

        pred_test = client_model.predict(X_test_retain, verbose=0)
        srv_test = tf.nn.softmax(server_model.predict(pred_test), axis=1).numpy()
        test_acc = accuracy_from_preds(srv_test, y_test_retain)
        conf_test = np.max(srv_test, axis=1)

        mask_c = np.isin(id_forget, client_ids[cid])
        if mask_c.sum() > 0:
            X_f = X_forget[mask_c]
            y_f = y_forget[mask_c]
            pred_f = client_model.predict(X_f, verbose=0)
            srv_f = tf.nn.softmax(server_model.predict(pred_f), axis=1).numpy()
            forget_acc = accuracy_from_preds(srv_f, y_f)
            conf_f = np.max(srv_f, axis=1)
        else:
            forget_acc = np.nan
            conf_f = np.array([])

        # 2) Simple MIA
        def simple_mia(scores_pos, scores_neg):
            """
            Simple membership‐inference attack with balanced positive and negative sets.
            Ensures equal number of member (scores_pos) and non‐member (scores_neg)
            samples for fair evaluation.
            Returns dict with keys {'auc','acc','prec','rec'}.
            """
            # If either set is empty, cannot compute meaningful metrics
            if len(scores_pos) == 0 or len(scores_neg) == 0:
                return {'auc': np.nan, 'acc': np.nan, 'prec': np.nan, 'rec': np.nan}
        
            # Determine how many to sample
            n = min(len(scores_pos), len(scores_neg))
        
            # Down‐sample each if necessary
            if len(scores_pos) > n:
                scores_pos = np.random.choice(scores_pos, n, replace=False)
            if len(scores_neg) > n:
                scores_neg = np.random.choice(scores_neg, n, replace=False)
        
            # Build labels and combined score vector
            labels = np.concatenate([np.ones(n), np.zeros(n)])
            scores = np.concatenate([scores_pos, scores_neg])
        
            # Compute predictions at threshold 0.5
            preds = (scores >= 0.5).astype(int)
        
            # Compute metrics
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = np.nan
        
            acc  = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec  = recall_score(labels, preds, zero_division=0)
        
            return {'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec}

        mia_train = simple_mia(conf_train, conf_test)
        mia_forget = simple_mia(conf_f, conf_test)
        mia_test = simple_mia(conf_test, conf_train)

        # 3) Shadow attack MIA on forget vs test
        try:
            # mask for this client's forgotten samples
            mask_c = np.isin(id_forget, client_ids[cid])
            X_pos = X_forget[mask_c]
            y_pos = y_forget[mask_c]
            # run the K-fold shadow attack on (forgotten vs full test)
            shadow_mia = compute_shadow_mia_metrics(
                client_model, server_model,
                client_data[cid],
                client_labels[cid],
                X_pos,              # positives: forgotten data
                X_test,        # negatives: FULL test set
                K=16,
                epochs_ft=10,
                batch_size=32
            )
        except Exception as e:
            tf.print(f"[WARN] Shadow-MIA failed for client {cid}: {e}")
            shadow_mia = {'auc':np.nan,'acc':np.nan,'prec':np.nan,'rec':np.nan}

       

        # 4) Backdoor
        bd = (evaluate_backdoor_attack_single_client(
                server_model, client_model, X_test,
                enable_backdoor=enable_backdoor,
                pattern=TRIGGER_PATTERN,
                trigger_label=TRIGGER_LABEL
            ) if enable_backdoor else 0.0)

        results.append({
            'train_acc': train_acc,
            'forget_acc': forget_acc,
            'test_acc': test_acc,
            # simple MIA
            **{f"mia_train_{k}":v for k,v in mia_train.items()},
            **{f"mia_forget_{k}":v for k,v in mia_forget.items()},
            **{f"mia_test_{k}":v for k,v in mia_test.items()},
            # shadow MIA
            **{f"mia_shadow_{k}":v for k,v in shadow_mia.items()},
            'backdoor': bd
        })

    # Global summary
    df = pd.DataFrame(results)
    gm = df.mean(numeric_only=True)

    print("\n=== Global averaged metrics ===", flush=True)
    print(f"Train Acc              : {gm.train_acc:.4f}", flush=True)
    print(f"Forget Acc             : {gm.forget_acc:.4f}", flush=True)
    print(f"Test Acc               : {gm.test_acc:.4f}\n", flush=True)

    def print_mia(label, prefix):
        print(f"– {label:^30} –")
        print(f"  AUC   : {gm[f'mia_{prefix}_auc']:.4f}", flush=True)
        print(f"  Acc   : {gm[f'mia_{prefix}_acc']:.4f}", flush=True)
        print(f"  Prec  : {gm[f'mia_{prefix}_prec']:.4f}", flush=True)
        print(f"  Rec   : {gm[f'mia_{prefix}_rec']:.4f}\n", flush=True)

    print_mia("Simple MIA: train vs test", "train")
    print_mia("Simple MIA: forget vs test", "forget")
    print_mia("Simple MIA: test vs train", "test")
    print_mia("Shadow MIA: forget vs test", "shadow")

    print(f"Backdoor Success Rate   : {gm.backdoor:.4f}", flush=True)
    print("================================\n", flush=True)

    return results


# =====================================================
# Helper: Dump all the data
# =====================================================
def _dump_csv(data, png_path):
    df = pd.DataFrame(data)
    csv_path = png_path.replace(".png", "_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")


# =====================================================
# Helper: View backdoor attack
# =====================================================
def save_backdoor_samples(X_forget, X_bd, out_dir):
    """
    Save a side-by-side PNG showing:
      - a sample from the poisoned training set
      - the corresponding sample used for backdoor eval
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # Poisoned training sample
    axes[0].imshow(X_forget[0])
    axes[0].axis('off')
    axes[0].set_title('Poisoned Training')
    # Backdoor-eval sample
    axes[1].imshow(X_bd[0])
    axes[1].axis('off')
    axes[1].set_title('Backdoor Test Input')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'backdoor_sample.png')
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved backdoor example to {out_path}")
# =====================================================
# Helper: Plot Forgotten Data Distribution Across Clients (Pie Chart)
# =====================================================
def plot_forgotten_distribution(client_ids, id_forget):
    forgotten_set = set(id_forget.tolist()) if isinstance(id_forget, np.ndarray) else set(id_forget)
    distribution = {}
    for i, ids in enumerate(client_ids):
        client_set = set(ids.tolist()) if isinstance(ids, np.ndarray) else set(ids)
        intersection = client_set.intersection(forgotten_set)
        distribution[i] = {
            "count": len(intersection),
            "ids": sorted(intersection)
        }
    for i, info in distribution.items():
        print(f"Client {i} has {info['count']} forgotten data points: {info['ids']}", flush=True)
    counts = [info['count'] for i, info in distribution.items()]
    labels = [f"Client {i}" for i in distribution.keys()]
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Forgotten Data Points per Client")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"forgotten_distribution_{timestamp}.png"
    filepath = os.path.join(GLOBAL_AVERAGE_FOLDER, file_name)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved forgotten distribution plot as {filepath}", flush=True)
    return distribution

# =====================================================
# Data Partitioning: Forgetting Set and Client Distribution
# =====================================================
def create_forgetting_set(
        X_train, y_train, id_train,
        forget_ratio,                     # <- required (or give a default)
        forget_class_list=None,
        forget_class_count=None,
        enable_backdoor: bool = True,     # <- name : type = default
        trigger_label=TRIGGER_LABEL,
        pattern=TRIGGER_PATTERN,
        trigger_mode=TRIGGER_MODE, trigger_value=TRIGGER_VALUE,
        trigger_color=TRIGGER_COLOR, trigger_patch_size=TRIGGER_PATCH_SIZE):

    # --- decide which samples to forget ------------------------------------
    y_int = np.argmax(y_train, axis=1)  # y is one-hot

    if forget_class_list is not None:
        cls = [c for c in cls if c != trigger_label]
        mask_to_forget = np.isin(y_int, cls)
    elif forget_class_count is not None:
        all_cls = np.unique(y_int)
        all_cls = all_cls[all_cls != trigger_label]
        chosen = np.random.choice(all_cls, forget_class_count, replace=False)
        mask_to_forget = np.isin(y_int, chosen)
    else:  # percentage-based forgetting (old behaviour)
        if forget_ratio is None:
            raise ValueError("forget_ratio is None and no class flags given.")
        num_forget = int(len(X_train) * forget_ratio)
        mask_to_forget = np.zeros(len(X_train), dtype=bool)
        mask_to_forget[np.random.choice(len(X_train), num_forget, replace=False)] = True

    forget_indices = np.where(mask_to_forget)[0]
    id_forget = id_train[forget_indices]
    forgotten_classes = np.unique(y_int[forget_indices]).tolist()

    # INSERT BACKDOOR into the selected forget set
    if enable_backdoor:
        X_forget_bd, y_forget_bd = insert_backdoor_trigger(
            X_train[forget_indices], y_train[forget_indices],
            trigger_label=trigger_label,
            patch_size=trigger_patch_size, trigger_value=trigger_value,
            pattern=pattern,
            trigger_mode=trigger_mode,
            trigger_color=trigger_color)
    else:
        X_forget_bd = X_train[forget_indices].copy()
        y_forget_bd = y_train[forget_indices].copy()

    # overwrite originals as before
    for i, idx in enumerate(forget_indices):
        X_train[idx] = X_forget_bd[i]
        y_train[idx] = y_forget_bd[i]
    

    # Prepare retraining set (excluding forget IDs)
    mask = np.isin(id_train, id_forget, invert=True)
    retrain_X = X_train[mask]
    retrain_y = y_train[mask]
    ids_train_kept = id_train[mask]

    if enable_backdoor:
        bd_count_initial = sum(
            has_backdoor(X_train[i], pattern, trigger_patch_size,
                         trigger_mode=trigger_mode,
                         trigger_color=trigger_color,
                         trigger_value=trigger_value)
            for i in range(len(X_train)))
        bd_count_retrain = sum(
            has_backdoor(retrain_X[i], pattern, trigger_patch_size,
                         trigger_mode=trigger_mode,
                         trigger_color=trigger_color,
                         trigger_value=trigger_value)
            for i in range(len(retrain_X)))
        print(f"Backdoor count in initial set: {bd_count_initial}")
        print(f"Backdoor count in retrained set: {bd_count_retrain}")
        assert bd_count_retrain == 0, "Backdoor leak detected!"

    return (X_forget_bd, y_forget_bd, id_forget, forgotten_classes), \
           (retrain_X, retrain_y, ids_train_kept)

def distribute_across_clients(X_train, y_train, ids_train, n_clients=N_CLIENTS, dirichlet_alpha=DIRCHLET_ALPHA):
    if dirichlet_alpha is not None:
        num_classes = y_train.shape[1]
        y_integers = np.argmax(y_train, axis=1)
        client_indices = {i: [] for i in range(n_clients)}
        for c in range(num_classes):
            idx_c = np.where(y_integers == c)[0]
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(alpha=np.repeat(dirichlet_alpha, n_clients))
            proportions = (proportions * len(idx_c)).astype(int)
            diff = len(idx_c) - np.sum(proportions)
            proportions[-1] += diff
            start = 0
            for i in range(n_clients):
                num_samples = proportions[i]
                client_indices[i].extend(idx_c[start:start+num_samples])
                start += num_samples
        client_data, client_labels, client_ids = [], [], []
        for i in range(n_clients):
            indices = sorted(client_indices[i])
            client_data.append(X_train[indices])
            client_labels.append(y_train[indices])
            client_ids.append(np.array(ids_train[indices]))
        return client_data, client_labels, client_ids
    else:
        split_size = len(X_train) // n_clients
        client_data, client_labels, client_ids = [], [], []
        for i in range(n_clients):
            start = i * split_size
            end = (i + 1) * split_size if i < n_clients - 1 else len(X_train)
            client_data.append(X_train[start:end])
            client_labels.append(y_train[start:end])
            client_ids.append(np.array(ids_train[start:end]))
        return client_data, client_labels, client_ids

# =====================================================
# Plotting Functions for Metrics
# =====================================================
def annotate_bars(ax, bars):
    def _safe_text(x, y, s, **kwargs):
        try:
            ax.text(x, y, s, **kwargs)
        except Exception as e:
            print(f"[WARN] Annotation failed: {e}", flush=True)

    if isinstance(bars, list):
        for container in bars:
            for bar in container:
                height = bar.get_height()
                _safe_text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f"{height:.3f}",
                    ha='center', va='bottom', fontsize=8
                )
    else:
        for bar in bars:
            height = bar.get_height()
            _safe_text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.3f}",
                ha='center', va='bottom', fontsize=8
            )

def plot_four_part_figure_for_client(
    phase_client_results: dict,
    phase_order_list: list[str],
    client_id: int,
    outfile: str
):
    """
    Creates a 2 × 2 figure for a single client with:
        ▸ Train / Forget / Test accuracy
        ▸ Four AUC curves: MIA‑Train, MIA‑Forget, MIA‑Test, Shadow‑MIA
        ▸ Δ‑Test‑accuracy vs. initial
        ▸ Back‑door success rate
    """
    # -------- gather data --------------------------------------------------
    train_acc, forget_acc, test_acc      = [], [], []
    mia_train, mia_forget, mia_test      = [], [], []
    mia_shadow                           = []
    backdoor                             = []

    for phase in phase_order_list:
        stats = phase_client_results[phase][client_id]
        train_acc  .append(stats["train_acc"])
        forget_acc .append(stats["forget_acc"])
        test_acc   .append(stats["test_acc"])
        mia_train  .append(stats["mia_train_auc"])
        mia_forget .append(stats["mia_forget_auc"])
        mia_test   .append(stats["mia_test_auc"])
        mia_shadow .append(stats["mia_shadow_auc"])
        backdoor   .append(stats["backdoor"])

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(phase_order_list))

    # ── 1) accuracy --------------------------------------------------------
    width = 0.25
    ax = axs[0, 0]
    bars1 = ax.bar(x - width, train_acc,  width, label="Train")
    bars2 = ax.bar(x,         forget_acc, width, label="Forget")
    bars3 = ax.bar(x + width, test_acc,   width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.set_title(f"Client {client_id} – accuracy")
    ax.legend()
    annotate_bars(ax, [bars1, bars2, bars3])

    # ── 2) MIA AUCs --------------------------------------------------------
    width = 0.20
    ax = axs[0, 1]
    bars1 = ax.bar(x - 1.5*width, mia_train,  width, label="MIA‑Train")
    bars2 = ax.bar(x - 0.5*width, mia_forget, width, label="MIA‑Forget")
    bars3 = ax.bar(x + 0.5*width, mia_test,   width, label="MIA‑Test")
    bars4 = ax.bar(x + 1.5*width, mia_shadow, width, label="Shadow‑MIA")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.set_title(f"Client {client_id} – membership‑inference AUC")
    ax.legend()
    annotate_bars(ax, [bars1, bars2, bars3, bars4])

    # ── 3) Δ‑test accuracy -------------------------------------------------
    ax = axs[1, 0]
    initial_test = test_acc[0]
    delta_test   = [t - initial_test for t in test_acc][1:]
    bars = ax.bar(np.arange(len(delta_test)), delta_test, color='orange')
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xticks(np.arange(len(delta_test)))
    ax.set_xticklabels(phase_order_list[1:], rotation=45, ha='right')
    ax.set_title(f"Client {client_id} – Δ‑test‑accuracy vs. initial")
    annotate_bars(ax, bars)

    # ── 4) back‑door success ----------------------------------------------
    ax = axs[1, 1]
    bars = ax.bar(x, backdoor, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.set_title(f"Client {client_id} – back‑door success")
    annotate_bars(ax, bars)

    # -------- save ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Client {client_id}] four‑part summary → {outfile}", flush=True)




def plot_four_part_figure_for_global(
    phase_client_results: dict,
    phase_order_list: list[str],
    outfile: str
):
    """
    2 × 2 figure averaged over all clients with:
        ▸ Train / Forget / Test accuracy
        ▸ Four AUC curves: MIA‑Train / Forget / Test / Shadow
        ▸ Δ‑Test‑accuracy vs initial
        ▸ Back‑door success
    """
    import numpy as np, matplotlib.pyplot as plt

    n_phases  = len(phase_order_list)
    n_clients = len(phase_client_results[phase_order_list[0]])

    # aggregate
    agg = {
        "train_acc":  np.zeros(n_phases),
        "forget_acc": np.zeros(n_phases),
        "test_acc":   np.zeros(n_phases),
        "mia_train":  np.zeros(n_phases),
        "mia_forget": np.zeros(n_phases),
        "mia_test":   np.zeros(n_phases),
        "mia_shadow": np.zeros(n_phases),
        "backdoor":   np.zeros(n_phases),
    }

    for p_idx, phase in enumerate(phase_order_list):
        for cid in range(n_clients):
            st = phase_client_results[phase][cid]
            agg["train_acc"][p_idx]  += st["train_acc"]
            agg["forget_acc"][p_idx] += st["forget_acc"]
            agg["test_acc"][p_idx]   += st["test_acc"]
            agg["mia_train"][p_idx]  += st["mia_train_auc"]
            agg["mia_forget"][p_idx] += st["mia_forget_auc"]
            agg["mia_test"][p_idx]   += st["mia_test_auc"]
            agg["mia_shadow"][p_idx] += st["mia_shadow_auc"]
            agg["backdoor"][p_idx]   += st["backdoor"]
        # mean over clients
        for k in agg.keys():
            agg[k][p_idx] /= n_clients

    # -------- figure -------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(n_phases)

    # 1) accuracies
    width = 0.25
    ax = axs[0, 0]
    b1 = ax.bar(x - width, agg["train_acc"],  width, label="Train")
    b2 = ax.bar(x,         agg["forget_acc"], width, label="Forget")
    b3 = ax.bar(x + width, agg["test_acc"],   width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.set_title("GLOBAL – accuracy")
    ax.legend()
    annotate_bars(ax, [b1, b2, b3])

    # 2) MIA AUCs
    width = 0.20
    ax = axs[0, 1]
    b1 = ax.bar(x - 1.5*width, agg["mia_train"],  width, label="MIA‑Train")
    b2 = ax.bar(x - 0.5*width, agg["mia_forget"], width, label="MIA‑Forget")
    b3 = ax.bar(x + 0.5*width, agg["mia_test"],   width, label="MIA‑Test")
    b4 = ax.bar(x + 1.5*width, agg["mia_shadow"], width, label="Shadow‑MIA")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.set_title("GLOBAL – membership‑inference AUC")
    ax.legend()
    annotate_bars(ax, [b1, b2, b3, b4])

    # 3) Δ‑test accuracy
    ax = axs[1, 0]
    delta_test = agg["test_acc"] - agg["test_acc"][0]
    bars = ax.bar(x, delta_test, color='orange')
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_title("GLOBAL – Δ‑test‑accuracy vs. initial")
    annotate_bars(ax, bars)

    # 4) back‑door
    ax = axs[1, 1]
    bars = ax.bar(x, agg["backdoor"], color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order_list, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.set_title("GLOBAL – back‑door success")
    annotate_bars(ax, bars)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global four‑part summary → {outfile}", flush=True)

def plot_global_metric_all_stages(phase_client_results, phase_order_list, metric, outfile):
    n_phases = len(phase_order_list)
    n_clients = len(phase_client_results[phase_order_list[0]])
    data = []
    for cid in range(n_clients):
        client_vals = []
        for phase in phase_order_list:
            client_vals.append(phase_client_results[phase][cid][metric])
        data.append(client_vals)
    data_arr = np.array(data)
    x = np.arange(n_clients)
    try:
        width = 0.8 / n_phases
        plt.figure(figsize=(10, 6))
        for phase_idx in range(n_phases):
            offset = phase_idx * width - 0.4
            plt.bar(x + offset, data_arr[:, phase_idx], width, label=f"{phase_order_list[phase_idx]}")
        plt.xlabel("Client")
        plt.ylabel(metric)
        plt.title(f"Per-Client Comparison: {metric}")
        plt.xticks(x, [f"Client {cid}" for cid in range(n_clients)], rotation=0)
        plt.ylim(0, 1)
        plt.legend()
        ax = plt.gca()
        annotate_bars(ax, ax.patches)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Global client comparison chart for metric '{metric}' saved to {outfile}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
    finally:
        txtfile = outfile.rsplit('.', 1)[0] + ".txt"
        with open(txtfile, "w") as f:
            header = "Client\t" + "\t".join(phase_order_list)
            f.write(header + "\n")
            for cid in range(n_clients):
                line = f"Client {cid}\t" + "\t".join([f"{data_arr[cid, phase_idx]:.3f}" for phase_idx in range(n_phases)])
                f.write(line + "\n")
        print(f"Global client comparison data for metric '{metric}' saved to {txtfile}")
        # flatten for CSV
        csv_dict = {"client": [f"client_{cid}" for cid in range(n_clients)]}
        for i, phase in enumerate(phase_order_list):
            csv_dict[phase] = data_arr[:, i]
        _dump_csv(csv_dict, outfile)

# =====================================================
# New Graphing Functions: Client Figures (one chart per image)
# =====================================================

def plot_client_accuracy(phase_client_results, phase_order_list, client_id, outfile):
    # Extract data
    train_acc, forget_acc, test_acc = [], [], []
    for phase in phase_order_list:
        stats = phase_client_results[phase][client_id]
        train_acc.append(stats["train_acc"])
        forget_acc.append(stats["forget_acc"])
        test_acc.append(stats["test_acc"])
    
    x = np.arange(len(phase_order_list))
    _dump_csv({
        "phase": phase_order_list,
        "train_acc": train_acc,
        "forget_acc": forget_acc,
        "test_acc": test_acc,
    }, outfile)
    try: 
        width = 0.25
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, train_acc, width, label="Train")
        plt.bar(x, forget_acc, width, label="Forget")
        plt.bar(x + width, test_acc, width, label="Test")
        plt.xticks(x, phase_order_list, rotation=45, ha='right', fontsize=10)
        plt.ylim([0,1])
        plt.xlabel("Experiment Phase", fontsize=12)
        plt.ylabel("Model Accuracy", fontsize=12)
        plt.title(f"Client {client_id} - Model Accuracy Comparison", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
    

def _collect_mia_metric(phase_client_results, phase_order_list, client_id, comp_key, metric):
    """
    comp_key ∈ {'mia_train','mia_forget','mia_test'}
    metric   ∈ {'auc','acc','prec','rec'}
    """
    return [
        phase_client_results[phase][client_id][f"{comp_key}_{metric}"]
        for phase in phase_order_list
    ]


def plot_client_mia(phase_client_results, phase_order_list, client_id, outfile_prefix):
    """
    Creates separate charts per metric and component combination:
      – mia_train_{metric}
      – mia_forget_{metric}
      – mia_test_{metric}
      – mia_shadow_{metric}
    and saves separate PNG and CSV files for each combination.
    Also creates combined plots with all components together.
    """
    
    comps = ("mia_train", "mia_forget", "mia_test", "mia_shadow")
    comp_labels = ("Train", "Forget", "Test", "Shadow")
    metrics = ("auc", "acc", "prec", "rec")
    
    # Helper function to collect values for a specific metric across all phases
    def collect_mia_metric(results, phases, cid, comp, metric):
        values = []
        for phase in phases:
            try:
                key = f"{comp}_{metric}"
                if key in results[phase][cid]:
                    values.append(results[phase][cid][key])
                else:
                    print(f"Warning: Missing key {key} for client {cid} in phase {phase}")
                    values.append(np.nan)
            except KeyError:
                print(f"Warning: Error accessing data for client {cid} in phase {phase}")
                values.append(np.nan)
        return values
    
    # Create separate plots for each component and metric combination
    for c_idx, comp in enumerate(comps):
        comp_label = comp_labels[c_idx]
        
        for metric in metrics:
            # Collect data for this component and metric across all phases
            data = collect_mia_metric(phase_client_results, phase_order_list, client_id, comp, metric)
            
            # Skip if all values are NaN
            if np.all(np.isnan(data)):
                print(f"Skipping plot for {comp}_{metric} as no data is available")
                continue
                
            # Create CSV
            csv_dict = {
                "phase": phase_order_list,
                f"{comp_label}_{metric}": data
            }
            
            # Save to CSV
            csv_filename = f"{outfile_prefix}_{comp}_{metric}.csv"
            pd.DataFrame(csv_dict).to_csv(csv_filename, index=False)
            print(f"Saved {csv_filename}")
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.bar(np.arange(len(phase_order_list)), data, width=0.6)
            plt.xticks(np.arange(len(phase_order_list)), phase_order_list, rotation=45, ha='right')
            plt.ylim([0, 1])
            plt.ylabel(f"{comp_label} MIA {metric.upper()}")
            plt.title(f"Client {client_id} – {comp_label} MIA {metric.upper()}")
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{outfile_prefix}_{comp}_{metric}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_filename}")
    
    # Also create the original combined plots for each metric
    x = np.arange(len(phase_order_list))
    width = 0.2  # width of the bars
    
    for metric in metrics:
        # Collect data for all components for this metric
        vals = [collect_mia_metric(phase_client_results, phase_order_list, client_id, comp, metric) 
                for comp in comps]
        
        # Check if we have any valid data
        has_data = False
        for val_list in vals:
            if not np.all(np.isnan(val_list)):
                has_data = True
                break
                
        if not has_data:
            print(f"Skipping combined plot for {metric} as no data is available")
            continue
        
        # Create CSV for combined data
        csv_dict = {"phase": phase_order_list}
        for lbl, v in zip(comp_labels, vals):
            csv_dict[f"{lbl.lower()}_{metric}"] = v
        
        # Save combined CSV
        combined_csv = f"{outfile_prefix}_combined_{metric}.csv"
        pd.DataFrame(csv_dict).to_csv(combined_csv, index=False)
        print(f"Saved {combined_csv}")
        
        # Create combined plot
        plt.figure(figsize=(12, 6))
        for i, (v, lbl) in enumerate(zip(vals, comp_labels)):
            # Only plot if we have some non-NaN data
            if not np.all(np.isnan(v)):
                plt.bar(x + (i - 1.5) * width, v, width, label=lbl)
        
        plt.xticks(x, phase_order_list, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel(f"MIA {metric.upper()}")
        plt.title(f"Client {client_id} – Combined MIA {metric.upper()}")
        plt.legend()
        plt.tight_layout()
        
        # Save combined plot
        combined_plot = f"{outfile_prefix}_combined_{metric}.png"
        plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {combined_plot}")
    

def plot_client_test_change(phase_client_results, phase_order_list, client_id, outfile):
    test_acc = []
    for phase in phase_order_list:
        stats = phase_client_results[phase][client_id]
        test_acc.append(stats["test_acc"])
    initial_test = test_acc[0]
    test_change = [t - initial_test for t in test_acc][1:]
    phases = phase_order_list[1:]
    
    x = np.arange(len(phases))
    _dump_csv({
        "phase": phases,
        "delta_test_acc": test_change,
    }, outfile)
    try:
        plt.figure(figsize=(12, 6))
        plt.bar(x, test_change, color='orange')
        plt.xticks(x, phases, rotation=45, ha='right', fontsize=10)
        plt.xlabel("Experiment Phase", fontsize=12)
        plt.ylabel("Test Accuracy Change", fontsize=12)
        plt.title(f"Client {client_id} - Test Accuracy Change vs Initial", fontsize=14)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
   

def plot_client_backdoor(phase_client_results, phase_order_list, client_id, outfile):
    backdoor = []
    for phase in phase_order_list:
        stats = phase_client_results[phase][client_id]
        backdoor.append(stats["backdoor"])
    
    x = np.arange(len(phase_order_list))
    _dump_csv({
        "phase": phase_order_list,
        "backdoor_success": backdoor,
    }, outfile)
    try:
        plt.figure(figsize=(12, 6))
        plt.bar(x, backdoor, color='red')
        plt.xticks(x, phase_order_list, rotation=45, ha='right', fontsize=10)
        plt.ylim([0,1])
        plt.xlabel("Experiment Phase", fontsize=12)
        plt.ylabel("Backdoor Attack Success", fontsize=12)
        plt.title(f"Client {client_id} - Backdoor Attack Success", fontsize=14)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
    

# =====================================================
# New Graphing Functions: Global Figures (one chart per image)
# =====================================================

def plot_global_accuracy(phase_client_results, phase_order_list, outfile):
    n_phases = len(phase_order_list)
    n_clients = len(phase_client_results[phase_order_list[0]])
    train_acc   = np.zeros(n_phases)
    forget_acc  = np.zeros(n_phases)
    test_acc    = np.zeros(n_phases)
    for i, phase in enumerate(phase_order_list):
        for cid in range(n_clients):
            stats = phase_client_results[phase][cid]
            train_acc[i] += stats["train_acc"]
            forget_acc[i] += stats["forget_acc"]
            test_acc[i] += stats["test_acc"]
        train_acc[i] /= n_clients
        forget_acc[i] /= n_clients
        test_acc[i] /= n_clients

    x = np.arange(n_phases)
    _dump_csv({
        "phase": phase_order_list,
        "train_acc": train_acc,
        "forget_acc": forget_acc,
        "test_acc": test_acc,
    }, outfile)
    try:
        width = 0.25
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, train_acc, width, label="Train")
        plt.bar(x, forget_acc, width, label="Forget")
        plt.bar(x + width, test_acc, width, label="Test")
        plt.xticks(x, phase_order_list, rotation=45, ha='right', fontsize=10)
        plt.xlabel("Experiment Phase", fontsize=12)
        plt.ylabel("Model Accuracy", fontsize=12)
        plt.title("Global - Model Accuracy Comparison", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
    

def plot_global_mia(phase_client_results, phase_order_list, outfile_prefix):
    """
    Creates separate charts per metric and component combination:
      – mia_train_{metric}
      – mia_forget_{metric}
      – mia_test_{metric}
      – mia_shadow_{metric}
    and saves separate PNG and CSV files for each combination.
    """
   
    
    n_phases = len(phase_order_list)
    n_clients = len(phase_client_results[phase_order_list[0]])
    comps = ("mia_train", "mia_forget", "mia_test", "mia_shadow")
    comp_labels = ("Train", "Forget", "Test", "Shadow")
    metrics = ("auc", "acc", "prec", "rec")
    
    # Create separate plots for each component and metric combination
    for c_idx, comp in enumerate(comps):
        comp_label = comp_labels[c_idx]
        
        for metric in metrics:
            key = f"{comp}_{metric}"
            
            # Collect data for this component and metric across all phases
            data = []
            for phase in phase_order_list:
                try:
                    # Get values for all clients for this phase, component, and metric
                    values = [
                        phase_client_results[phase][cid][key]
                        for cid in range(n_clients)
                        if key in phase_client_results[phase][cid]
                    ]
                    
                    if values:
                        data.append(np.mean(values))
                    else:
                        print(f"Warning: No data found for {key} in phase {phase}")
                        data.append(np.nan)
                except KeyError:
                    print(f"Warning: Missing key {key} in phase {phase}")
                    data.append(np.nan)
            
            # Skip if all values are NaN
            if np.all(np.isnan(data)):
                print(f"Skipping plot for {key} as no data is available")
                continue
                
            # Create CSV
            csv_dict = {
                "phase": phase_order_list,
                f"{comp_label}_{metric}": data
            }
            
            # Save to CSV
            csv_filename = f"{outfile_prefix}_{comp}_{metric}.csv"
            pd.DataFrame(csv_dict).to_csv(csv_filename, index=False)
            print(f"Saved {csv_filename}")
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.bar(np.arange(n_phases), data, width=0.6)
            plt.xticks(np.arange(n_phases), phase_order_list, rotation=45, ha='right')
            plt.ylim([0, 1])
            plt.ylabel(f"{comp_label} MIA {metric.upper()}")
            plt.title(f"Global – {comp_label} MIA {metric.upper()}")
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{outfile_prefix}_{comp}_{metric}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_filename}")
    
    # Also create the original combined plots for each metric
    x = np.arange(n_phases)
    width = 0.2  # width of the bars
    
    for metric in metrics:
        # Prepare data for this metric across all components and phases
        data = np.zeros((len(comps), n_phases))
        has_data = False
        
        for c_idx, comp in enumerate(comps):
            key = f"{comp}_{metric}"
            for p_idx, phase in enumerate(phase_order_list):
                try:
                    values = [
                        phase_client_results[phase][cid][key]
                        for cid in range(n_clients)
                        if key in phase_client_results[phase][cid]
                    ]
                    
                    if values:
                        data[c_idx, p_idx] = np.mean(values)
                        has_data = True
                    else:
                        data[c_idx, p_idx] = np.nan
                except KeyError:
                    data[c_idx, p_idx] = np.nan
        
        if not has_data:
            print(f"Skipping combined plot for {metric} as no data is available")
            continue
            
        # Create CSV for combined data
        csv_dict = {"phase": phase_order_list}
        for c_idx, comp_label in enumerate(comp_labels):
            csv_dict[f"{comp_label}_{metric}"] = data[c_idx]
        
        # Save combined CSV
        combined_csv = f"{outfile_prefix}_combined_{metric}.csv"
        pd.DataFrame(csv_dict).to_csv(combined_csv, index=False)
        print(f"Saved {combined_csv}")
        
        # Create combined plot
        plt.figure(figsize=(12, 6))
        for c_idx, comp_label in enumerate(comp_labels):
            # Only plot if we have some non-NaN data
            if not np.all(np.isnan(data[c_idx])):
                plt.bar(x + (c_idx-1.5)*width, data[c_idx], width, label=comp_label)
        
        plt.xticks(x, phase_order_list, rotation=45, ha='right')
        plt.ylim([0, 1])
        plt.ylabel(f"MIA {metric.upper()}")
        plt.title(f"Global – Combined MIA {metric.upper()}")
        plt.legend()
        plt.tight_layout()
        
        # Save combined plot
        combined_plot = f"{outfile_prefix}_combined_{metric}.png"
        plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {combined_plot}")


    

def plot_global_test_change(phase_client_results, phase_order_list, outfile):
    n_phases = len(phase_order_list)
    n_clients = len(phase_client_results[phase_order_list[0]])
    test_acc    = np.zeros(n_phases)
    for i, phase in enumerate(phase_order_list):
        for cid in range(n_clients):
            stats = phase_client_results[phase][cid]
            test_acc[i] += stats["test_acc"]
        test_acc[i] /= n_clients
    initial_test = test_acc[0]
    test_change = test_acc - initial_test
    _dump_csv({
        "phase": phase_order_list,
        "delta_test_acc": test_change,
    }, outfile)
    try:
        x = np.arange(n_phases)
        plt.figure(figsize=(12, 6))
        plt.bar(x, test_change, color='orange')
        plt.xticks(x, phase_order_list, rotation=45, ha='right', fontsize=10)
        plt.xlabel("Experiment Phase", fontsize=12)
        plt.ylabel("Test Accuracy Change", fontsize=12)
        plt.title("Global - Test Accuracy Change vs Initial", fontsize=14)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
    

def plot_global_backdoor(phase_client_results, phase_order_list, outfile):
    n_phases = len(phase_order_list)
    n_clients = len(phase_client_results[phase_order_list[0]])
    backdoor    = np.zeros(n_phases)
    for i, phase in enumerate(phase_order_list):
        for cid in range(n_clients):
            stats = phase_client_results[phase][cid]
            backdoor[i] += stats["backdoor"]
        backdoor[i] /= n_clients
    _dump_csv({
        "phase": phase_order_list,
        "backdoor_success": backdoor,
    }, outfile)
    try:
        x = np.arange(n_phases)
        plt.figure(figsize=(12, 6))
        plt.bar(x, backdoor, color='red')
        plt.xticks(x, phase_order_list, rotation=45, ha='right', fontsize=10)
        plt.xlabel("Experiment Phase", fontsize=12)
        plt.ylabel("Backdoor Attack Success", fontsize=12)
        plt.title("Global - Backdoor Attack Success", fontsize=14)
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)
    
    
def get_strategy_phases(phases, strategy):
    """
    Given the complete list of phases and a strategy name,
    return a list that contains "Initial", then all phases (except "Retrain") that start with the strategy,
    and finally "Retrain". This groups experiments by strategy.
    """
    selected = ["Initial"]
    for phase in phases:
        if phase not in ["Initial", "Retrain"] and phase.startswith(strategy):
            selected.append(phase)
    selected.append("Retrain")
    return selected
# ──────────────────────────────────────────────────────────
# per‑client plotting helpers
# ──────────────────────────────────────────────────────────
def aggregate_round_metrics(df: pd.DataFrame, epochs_per_round: int) -> pd.DataFrame:
    if "epoch" not in df.columns:
        raise KeyError("aggregate_round_metrics: no 'epoch' column in df")
    if epochs_per_round <= 0:
        raise ValueError("epochs_per_round must be positive")

    df_r = df.copy()
    df_r["round"] = ((df_r["epoch"] - 1) // epochs_per_round) + 1
    numeric_cols = df_r.select_dtypes(include="number").columns
    grouped = df_r.groupby("round", as_index=False)[numeric_cols].mean()
    grouped = grouped.rename(columns={"round": "epoch"})
    return grouped


def _plot_and_save(df: pd.DataFrame, x_key: str, y_keys: List[str], out_png: str, title: str):
    try:
        plt.figure(figsize=(6, 4))
        for k in y_keys:
            plt.plot(df[x_key], df[k], "-o", label=k)
        plt.xlabel("round")
        plt.ylabel("value")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved plot →", out_png, flush=True)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)


def save_history(
    hist: List[Dict[str, Any]],
    client_folder: str,
    phase_name: str,
    *,
    epochs_per_round: int
):
    if not hist:
        print(f"[warn] history empty – nothing to save for {phase_name}")
        return

    # ── epoch-level DataFrame & CSV ───────────────────────────────────────
    df_epoch = pd.DataFrame(hist)
    epoch_csv = os.path.join(client_folder, f"{phase_name}_epoch_metrics.csv")
    df_epoch.to_csv(epoch_csv, index=False)
    print("Saved epoch-level CSV →", epoch_csv)

    # ── epoch-level plot ─────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8,4))
        for col in df_epoch.columns:
            if col != "epoch":
                ax.plot(df_epoch["epoch"], df_epoch[col], marker="o", label=col)
        ax.set_xlabel("Epoch")
        ax.set_title(f"{phase_name} — per-epoch metrics")
        ax.legend(loc="best")
        plt.tight_layout()
        epoch_png = os.path.join(client_folder, f"{phase_name}_epoch_metrics.png")
        plt.savefig(epoch_png, dpi=300)
        plt.close(fig)
        print("Saved epoch-level plot →", epoch_png)
    except Exception as e:
        print(f"[WARN] Epoch-level plotting failed: {e}")

    # ── compute round-level aggregates ────────────────────────────────────
    df_round = df_epoch.copy()
    df_round = aggregate_round_metrics(df_epoch, epochs_per_round)
    df_round.index.name = "round"
    df_round.reset_index(inplace=True)
    # average numeric columns per round
    numeric = df_round.select_dtypes(include="number").columns.drop("round")
    df_round = df_round.groupby("round", as_index=False)[numeric].mean()

    round_csv = os.path.join(client_folder, f"{phase_name}_round_metrics.csv")
    df_round.to_csv(round_csv, index=False)
    print("Saved round-level CSV →", round_csv)

    # ── round-level plot ─────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8,4))
        for col in numeric:
            ax.plot(df_round["round"], df_round[col], marker="o", label=col)
        ax.set_xlabel("Round")
        ax.set_title(f"{phase_name} — per-round metrics")
        ax.legend(loc="best")
        plt.tight_layout()
        round_png = os.path.join(client_folder, f"{phase_name}_round_metrics.png")
        plt.savefig(round_png, dpi=300)
        plt.close(fig)
        print("Saved round-level plot →", round_png)
    except Exception as e:
        print(f"[WARN] Round-level plotting failed: {e}")



# ──────────────────────────────────────────────────────────
# global (averaged) helpers
# ──────────────────────────────────────────────────────────
def save_global_history(
    hist_list: List[List[Dict[str, Any]]],
    out_folder: str,
    phase_name: str,
    *,
    epochs_per_round: int):
    non_empty = [h for h in hist_list if h]
    if not non_empty:
        print(f"[warn] nothing to save for {phase_name}")
        return

    # Build & average epoch-level DataFrame
    df_epoch = pd.DataFrame(non_empty[0])
    df_epoch.loc[:, df_epoch.columns != "epoch"] = 0.0
    contributors = 0
    for h in non_empty:
        for col in df_epoch.columns:
            if col != "epoch":
                df_epoch[col] += [rec[col] for rec in h]
        contributors += 1
    df_epoch.loc[:, df_epoch.columns != "epoch"] /= contributors

    # Aggregate into rounds and rename the index column
    df_round = aggregate_round_metrics(df_epoch, epochs_per_round) \
                   .rename(columns={"epoch": "round"})
                   
    df_round = aggregate_round_metrics(df_epoch, epochs_per_round)
    df_round.index.name = "round"
    df_round.reset_index(inplace=True)
    csv_round = os.path.join(out_folder, f"{phase_name}_round_metrics_global.csv")
    df_round.to_csv(csv_round, index=False)

    # Plot GLOBAL per-round metrics (using the "round" column)
    try:
        png_round = csv_round.replace(".csv", ".png")
        _plot_and_save(
            df_round,
            "round",                                         # ← use "round" here
            [c for c in df_round.columns if c != "round"],
            png_round,
            f"{phase_name} – GLOBAL avg (per-round)"
        )
        print("Saved GLOBAL CSV →", csv_round)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)




# =====================================================
# Main Experimental Routine (Updated Plotting Section)
# =====================================================
def main():
    global BASE_EPOCHS, FT_EPOCHS, N_CLIENTS, BATCH_SIZE, LEARNING_RATE, FORGET_RATIO
    global TRIGGER_MODE, TRIGGER_COLOR, TRIGGER_LABEL, TRIGGER_PATTERN, TRIGGER_PATCH_SIZE, TRIGGER_VALUE
    global DIRCHLET_ALPHA, SUBSET_RATIO, EDGE_BOOST, USE_PCA
    global GLOBAL_AVERAGE_FOLDER, GLOBAL_CLIENT_FOLDERS
    global OUTPUT_FOLDER, WEIGHTING_SCHEME, DEVALUATION
    global FORGET_CLASS_LIST, FORGET_CLASS_COUNT
    global GRAD_DIR, CLIENT_BLOCKS, ENABLE_BACKDOOR, EPOCHS_PER_ROUND
    global OPTIMIZER, MOMENTUM, WEIGHT_DECAY, AUGMENT

    start_time = time.time()
    
    # ---------------------------
    # Parse Command-Line Arguments and Update Hyperparameters
    # ---------------------------
    args, selected_strats = parse_args()
    BASE_EPOCHS   = args.base_epochs
    FT_EPOCHS = args.ft_epochs if args.ft_epochs is not None else max(1, BASE_EPOCHS // 5)
    N_CLIENTS     = args.n_clients
    BATCH_SIZE    = args.batch_size
    LEARNING_RATE = args.learning_rate
    FORGET_RATIO  = args.forget_ratio
    TRIGGER_LABEL = args.trigger_label
    TRIGGER_PATTERN = args.trigger_pattern
    TRIGGER_PATCH_SIZE = args.trigger_patch_size
    TRIGGER_VALUE = args.trigger_value
    DIRCHLET_ALPHA = args.dirchlet_alpha  # Note: variable name remains as in your code.
    SUBSET_RATIO  = args.subset_ratio
    USE_PCA       = args.use_pca
    TRIGGER_MODE = args.trigger_mode
    TRIGGER_COLOR = args.trigger_color
    FORGET_CLASS_LIST  = args.forget_class_list
    FORGET_CLASS_COUNT = args.forget_class_count
    CLIENT_BLOCKS = args.client_blocks
    if args.forget_ratio is None:
      args.forget_ratio = 0.0
    ENABLE_BACKDOOR = not args.no_backdoor
    EPOCHS_PER_ROUND = args.epochs_per_round
    OPTIMIZER = args.optimizer
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay
    AUGMENT = args.augment
    


    # NEW: Number of iterative unlearning updates.
    iterated_unlearning = args.iterated_unlearning
    print(f"Iterated unlearning will be applied {iterated_unlearning} time(s)", flush=True)
    # =====================================================
    # Create Output Folder & Subfolders
    # =====================================================
    timestamp_folder = time.strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_FOLDER = f"Unlearn_Comparison_{timestamp_folder}"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    print("Main output folder created:", OUTPUT_FOLDER, flush=True)

    # Copy the source code file into the output folder (named as source_code.py)
    try:
        shutil.copy(__file__, os.path.join(OUTPUT_FOLDER, "source_code.py"))
        print("Source code copied to:", os.path.join(OUTPUT_FOLDER, "source_code.py"), flush=True)
    except Exception as e:
        print("Failed to copy source code file:", e, flush=True)

    GLOBAL_AVERAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "average")
    os.makedirs(GLOBAL_AVERAGE_FOLDER, exist_ok=True)
    GLOBAL_CLIENT_FOLDERS = {}
    for i in range(N_CLIENTS):
        client_folder = os.path.join(OUTPUT_FOLDER, f"client_{i}")
        os.makedirs(client_folder, exist_ok=True)
        GLOBAL_CLIENT_FOLDERS[i] = client_folder
        print(f"Created folder for Client {i}: {client_folder}", flush=True)
    print("Created 'average' folder:", GLOBAL_AVERAGE_FOLDER, flush=True)
   

    # ---------------------------
    # Set Up Output Folders
    # ---------------------------
    GLOBAL_AVERAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "global_results")
    os.makedirs(GLOBAL_AVERAGE_FOLDER, exist_ok=True)
    GLOBAL_CLIENT_FOLDERS = []
    for i in range(N_CLIENTS):
        client_folder = os.path.join(OUTPUT_FOLDER, f"client_{i}_results")
        os.makedirs(client_folder, exist_ok=True)
        GLOBAL_CLIENT_FOLDERS.append(client_folder)
    # ---------------------------
    # Gradients folder
    # ---------------------------
    GRAD_DIR = os.path.join(OUTPUT_FOLDER, "grad_shards")
    os.makedirs(GRAD_DIR, exist_ok=True)
    # ---------------------------
    # Data Loading and Preprocessing
    # ---------------------------
    dataset_choice = args.dataset.lower()
    if dataset_choice == "mnist":
        input_shape = (28, 28, 1)
        num_classes = 10
        X, y, X_test, y_test, _ = load_and_preprocess_mnist(subset_ratio=SUBSET_RATIO)
    elif dataset_choice == "cifar10":
        input_shape = (32, 32, 3)
        num_classes = 10
        X, y, X_test, y_test, _, datagen = load_and_preprocess_cifar10(subset_ratio=SUBSET_RATIO)
    elif dataset_choice == "ham10000":
        input_shape = (128, 128, 3)
        num_classes = 7
        X, y, X_test, y_test, datagen = load_and_preprocess_ham10000(subset_ratio=SUBSET_RATIO)
    else:
        print("Unsupported dataset selected. Exiting.", flush=True)
        sys.exit(1)

    X_train, X_test, y_train, y_test, id_train, id_test = split_data(X, y)
    X_test_full, y_test_full = X_test.copy(), y_test.copy()

    # Create forgetting set (inserts backdoors and returns indices of forgotten samples)
    (X_forget, y_forget, id_forget, forgotten_classes), \
   (retrain_X, retrain_y, ids_train_kept) = create_forgetting_set(
            X_train, y_train, id_train,
            forget_ratio       = FORGET_RATIO,
            forget_class_list  = FORGET_CLASS_LIST,
            forget_class_count = FORGET_CLASS_COUNT,
            trigger_label      = TRIGGER_LABEL,
            pattern            = TRIGGER_PATTERN,
            trigger_mode       = TRIGGER_MODE,
            trigger_patch_size      = TRIGGER_PATCH_SIZE, trigger_value=TRIGGER_VALUE,
            trigger_color      = TRIGGER_COLOR, enable_backdoor=ENABLE_BACKDOOR)

    # -------- NEW BLOCK ---------
    if args.forget_class_list is not None or args.forget_class_count is not None:
        y_test_int = np.argmax(y_test, axis=1)
        keep_mask  = ~np.isin(y_test_int, forgotten_classes)
        removed    = np.sum(~keep_mask)
        num_left   = np.sum(keep_mask)   # <-- compute this before using it
    
        if num_left == 0:
            raise ValueError(
                "Filtering removed every test sample — nothing left to validate on. "
                "Choose a smaller --forget_class_list / --forget_class_count."
            )
    
        X_test, y_test = X_test[keep_mask], y_test[keep_mask]
        print(f"Filtered {removed} test samples "
              f"belonging to forgotten classes {forgotten_classes}",flush=True)

    # hyperparameters
    hp_file = os.path.join(OUTPUT_FOLDER, "hyperparameters.txt")
    write_hyperparameters_file(
        hp_path=hp_file,
        selected_strats=selected_strats,
        args=args,
        client_blocks=CLIENT_BLOCKS,
        forgotten_classes=forgotten_classes  # if you computed this earlier
    )
    # ----------------------------
    # Replace the corresponding training examples with the backdoored ones.
    for i, uid in enumerate(id_forget):
        idx = np.where(id_train == uid)[0][0]
        X_train[idx] = X_forget[i]
        y_train[idx] = y_forget[i]
    print(f"Total training samples: {len(X_train)}", flush=True)
    print(f"Samples to forget: {len(X_forget)} ({len(X_forget)/len(X_train)*100:.2f}%)", flush=True)
    print(f"Retained samples for retraining: {len(retrain_X)}", flush=True)
    print(f"Test samples: {len(X_test)}", flush=True)

    # Create client partitions. (Correct keyword used: dirichlet_alpha)
    client_data, client_labels, client_ids = distribute_across_clients(X_train, y_train, id_train, N_CLIENTS, dirichlet_alpha=DIRCHLET_ALPHA)
    for i in range(N_CLIENTS):
        print(f"Client {i} (initial training) has {len(client_data[i])} samples", flush=True)

    # Create fine-tuning data (filtered client data without forgotten samples).
    client_data_ft, client_labels_ft, client_ids_ft = [], [], []
    for i in range(N_CLIENTS):
        mask = ~np.isin(client_ids[i], id_forget)
        filtered_data   = client_data[i][mask]
        filtered_labels = client_labels[i][mask]
        filtered_ids    = client_ids[i][mask]
        client_data_ft.append(filtered_data)
        client_labels_ft.append(filtered_labels)
        client_ids_ft.append(filtered_ids)
        removed_count = np.sum(np.isin(client_ids[i], id_forget))
        print(f"Client {i}: Removed {removed_count} forgotten samples; now has {filtered_ids.shape[0]}", flush=True)
    client_retain_data   = client_data_ft          # already computed
    client_retain_labels = client_labels_ft
    # Create test backdoor attacks for validation
    X_bd, _ = insert_backdoor_trigger(
    X_test.copy(), y_test.copy(),
    trigger_label=TRIGGER_LABEL,  # won't be used
    pattern=TRIGGER_PATTERN,
    patch_size=TRIGGER_PATCH_SIZE,
    trigger_value=TRIGGER_VALUE,
    trigger_mode=TRIGGER_MODE,
    trigger_color=TRIGGER_COLOR
    )
    save_backdoor_samples(X_forget, X_bd, OUTPUT_FOLDER)
    
    # ------------------------------------------------------
    # build per-client forget sets
    client_forget_data   = []
    client_forget_labels = []
    for i in range(N_CLIENTS):
        # pick out exactly the forget-set samples that belong to client i
        mask = np.isin(id_forget, client_ids[i])
        client_forget_data.append  (X_forget[mask])
        client_forget_labels.append(y_forget[mask])
        print(f"[Init] Client {i}: forget-set size = {len(client_forget_data[-1])}")
    # ------------------------------------------------------
    # ---------------------------
    # Mapping of Unlearning Strategies to Functions
    # ---------------------------
    strategy_func_map = {
        "NoOrth": aggregate_no_orth,
        "ProjRaw": aggregate_proj_raw,
        "PCA": aggregate_pca,
        "GS": aggregate_gs,
        "PCA+GS": aggregate_pca_gs,
        "NoUnlearn": aggregate_identity,
        "GS-no-proj": aggregate_gs_no_proj
    }

    # ---------------------------
    # Phase 1: Initial Training
    # ---------------------------
    print("\n--- Phase 1: Initial Training ---", flush=True)
    if args.optimizer=="sgd":
        def make_opt(lr):
            return tf.keras.optimizers.experimental.SGD(
                       learning_rate=lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay)
    elif args.optimizer=="adamw":
        def make_opt(lr):
            return tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=5e-4)
    else:
        def make_opt(lr):
            return tf.keras.optimizers.Adam(learning_rate=lr)
    clients_initial = [create_client_model(input_shape, CLIENT_BLOCKS)
                   for _ in range(N_CLIENTS)]
    temp_model = create_client_model(input_shape, CLIENT_BLOCKS)
    client_out_shape = temp_model(tf.random.uniform((1,)+input_shape)).shape[1:]
    server_initial = create_server_model(
        client_out_shape,
        num_classes,
        CLIENT_BLOCKS,
        optimizer_fn=make_opt,
        learning_rate=LEARNING_RATE
    )
   
    batch_map, server_cache, client_caches, hist_init = train_sequential(
        clients_initial,
        server_initial,
        client_data,
        client_labels,
        client_ids,
        optimizer_fn      = make_opt,
        datagen           = datagen if args.augment else None,
        X_val             = X_test,
        y_val             = y_test,
        X_bd              = X_bd,
        y_bd              = y_test,
        batch_size        = BATCH_SIZE,
        base_epochs       = BASE_EPOCHS,
        epochs_per_round  = EPOCHS_PER_ROUND,
        cache_gradients   = True,
        learning_rate     = LEARNING_RATE,
        forget_data   = client_forget_data,    # already added earlier
        forget_labels = client_forget_labels,
        retain_data   = client_retain_data,    # NEW
        retain_labels = client_retain_labels,  # NEW
    )

    for cid in range(N_CLIENTS):
        save_history(hist_init[cid], GLOBAL_CLIENT_FOLDERS[cid],"Initial", epochs_per_round=EPOCHS_PER_ROUND)
    save_global_history(hist_init,           
                          GLOBAL_AVERAGE_FOLDER,
                          "Initial", epochs_per_round=EPOCHS_PER_ROUND)

    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "id_train": id_train,
        "X_forget": X_forget,
        "y_forget": y_forget,
        "id_forget": id_forget,
        "retrain_X": retrain_X,
        "retrain_y": retrain_y,
        "ids_train_kept": ids_train_kept,
        "client_data": client_data,
        "client_labels": client_labels,
        "client_ids": client_ids,
        "client_data_ft": client_data_ft,
        "client_labels_ft": client_labels_ft,
        "client_ids_ft": client_ids_ft,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "X_test": X_test,
        "y_test": y_test,
        "batch_map": batch_map,
        "server_cache": server_cache,
        "client_caches": client_caches,
        "server_initial_weights": server_initial.get_weights(),
        "clients_initial_weights": [client.get_weights() for client in clients_initial]
    }

    phase_client_results = {}
    phase_client_results["Initial"] = gather_all_metrics(server_initial, clients_initial, client_data, client_labels, X_forget, y_forget, id_forget, client_ids, X_test_full, y_test_full, X_test, y_test, X_train, y_train, enable_backdoor=ENABLE_BACKDOOR)

    experiment_phase_keys = []  # Collect keys for experiments.

    # ---------------------------
    # Phase 2: Experiments (Unlearning & Fine-tuning)
    # ---------------------------
    for strat in selected_strats:
        strategy_func = strategy_func_map.get(strat)
        tf.keras.backend.clear_session()
        print("\n-Cleared session", flush=True)
        if strategy_func is None:
            print(f"Unknown unlearning strategy '{strat}', skipping...", flush=True)
            continue

        for ws, eb, dv in itertools.product(args.weighting_schemes, args.edge_boost, args.devaluation):
            # Use local copies of the experimental configuration
            current_scheme = ws
            current_edge_boost = eb
            current_devaluation = dv
            config_label = f"{strat}_WS-{current_scheme}_EB-{current_edge_boost}_DV-{current_devaluation}"
            print(f"\n*** Running experiment for configuration: {config_label} ***\n", flush=True)

            # Create experiment models.
            # ── 1.  fresh client models (same split as the initial run) ─────────────
            clients_exp = [
                create_client_model(input_shape, CLIENT_BLOCKS)     # ← pass split size
                for _ in range(N_CLIENTS)
            ]
            for i, cm in enumerate(clients_exp):
                cm.set_weights(data_dict["clients_initial_weights"][i])
            
            # ── 2.  fresh server model, also aware of the split ─────────────────────
            server_model_exp = create_server_model(
                client_out_shape,
                num_classes,
                CLIENT_BLOCKS,
                optimizer_fn=make_opt,
                learning_rate=LEARNING_RATE
            )
            server_model_exp.set_weights(data_dict["server_initial_weights"])
            
            # ── 3.  make experiment-local *dict* copies of the gradient caches ──────
            #       (the files themselves are shared, so other experiments can still
            #       read them even if this run deletes entries in its own copy)
            server_cache_exp  = dict(data_dict["server_cache"])
            client_caches_exp = [dict(cc) for cc in data_dict["client_caches"]]
            
            # ── 4.  compute the batch ids we need to unlearn ────────────────────────
            forget_batches = {
                b
                for uid in id_forget
                for b   in data_dict["batch_map"].get(uid, [])
            }

            # Apply iterative unlearning.
            for iter_idx in range(iterated_unlearning):
                print(f"Iterative unlearning pass {iter_idx+1}/{iterated_unlearning} for configuration: {config_label}", flush=True)
                server_model_exp = unlearn_server_with_strategy(server_model_exp, server_cache_exp, forget_batches, strategy_func, LEARNING_RATE, scheme=current_scheme, edge_boost=current_edge_boost, devaluation=current_devaluation)
                for cid in range(N_CLIENTS):
                    clients_exp[cid] = unlearn_client_with_strategy(clients_exp[cid], client_caches_exp[cid], cid, forget_batches, strategy_func, LEARNING_RATE, logging=False, scheme=current_scheme, edge_boost=current_edge_boost, devaluation=current_devaluation)

            phase_name_unlearn = f"{strat}-unlearn_WS-{current_scheme}_EB-{current_edge_boost}_DV-{current_devaluation}"
            phase_client_results[phase_name_unlearn] = gather_all_metrics(server_model_exp, clients_exp, client_data, client_labels, X_forget, y_forget,id_forget, client_ids, X_test_full, y_test_full, X_test, y_test, X_train, y_train, enable_backdoor=ENABLE_BACKDOOR)
            experiment_phase_keys.append(phase_name_unlearn)

            print(f"Fine-tuning for configuration: {config_label}", flush=True)
            client_data_ft_exp  = [np.copy(arr) for arr in client_data_ft]
            client_labels_ft_exp = [np.copy(arr) for arr in client_labels_ft]
            client_ids_ft_exp   = [np.copy(arr) for arr in client_ids_ft]
            _, _, _, hist_ft = train_sequential(
                clients_exp,
                server_model_exp,
                client_data_ft_exp,
                client_labels_ft_exp,
                client_ids_ft_exp,
                optimizer_fn      = make_opt,
                datagen           = datagen if args.augment else None,
                X_val             = X_test,
                y_val             = y_test,
                X_bd              = X_bd,
                y_bd              = y_test,
                batch_size        = BATCH_SIZE,
                base_epochs       = FT_EPOCHS,
                epochs_per_round  = EPOCHS_PER_ROUND,
                cache_gradients   = False,
                learning_rate     = LEARNING_RATE,
                forget_data   = client_forget_data,    # already added earlier
                forget_labels = client_forget_labels,
                retain_data   = client_retain_data,    # NEW
                retain_labels = client_retain_labels,  # NEW
            )

            phase_name_finetune = f"{strat}-finetune_WS-{current_scheme}_EB-{current_edge_boost}_DV-{current_devaluation}"
            for cid in range(N_CLIENTS):
                phase_csv = f"{phase_name_finetune}"          # same string you already use
                save_history(hist_ft[cid],
                                  GLOBAL_CLIENT_FOLDERS[cid],
                                  phase_csv, epochs_per_round=EPOCHS_PER_ROUND)
            save_global_history(hist_ft,            # <‑‑ NEW
                          GLOBAL_AVERAGE_FOLDER,
                           f"{phase_name_finetune}", epochs_per_round=EPOCHS_PER_ROUND)
           
            phase_client_results[phase_name_finetune] = gather_all_metrics(
                server_model_exp,
                clients_exp,
                client_data_ft_exp,
                client_labels_ft_exp,
                X_forget,           # X_forget  → X_forget
                y_forget,           # y_forget  → your one-hot labels for the forget set
                id_forget,          # id_forget → the array of IDs you want to forget
                client_ids,         # client_ids → the list of each client's full ID array
                X_test_full,             # X_test    → global test-set inputs
                y_test_full, X_test, y_test,              # y_test    → global test-set labels
                X_train,            # X_train   → global training inputs (for shadow/MIA)
                y_train,            # y_train   → global training labels
                enable_backdoor=ENABLE_BACKDOOR
            )

            experiment_phase_keys.append(phase_name_finetune)

    # ---------------------------
    # Cleanup: Delete Cached Gradient Files
    # ---------------------------
    # [CLEANUP] Removing top-level cached gradient shards...
    top_grad = "grad_shards"
    if os.path.exists(top_grad):
        shutil.rmtree(top_grad)
        print(f"[CLEANUP] Deleted gradient directory: {top_grad}", flush=True)
    else:
        print(f"[CLEANUP] Gradient directory does not exist: {top_grad}", flush=True)

    # ---------------------------
    # Phase 3: Retraining from Scratch
    # ---------------------------
    print("\n--- Phase 3: Retraining from Scratch ---", flush=True)
    
   
    tf.keras.backend.clear_session()
    print("\n-Cleared session", flush=True)
    # ── 1. build fresh client models for retraining ───────────────
    clients_retrain = [
        create_client_model(input_shape, CLIENT_BLOCKS)    # ← pass split size
        for _ in range(N_CLIENTS)
    ]
    
    # ── 2. recompute the output shape of the client half ──────────
    temp_model = create_client_model(input_shape, CLIENT_BLOCKS)
    client_out_shape = temp_model(
        tf.random.uniform((1,)+input_shape)
    ).shape[1:]                                            # (H, W, C)
    
    # ── 3. build the matching server half ─────────────────────────
    server_retrain = create_server_model(
        client_out_shape,
        num_classes,
        CLIENT_BLOCKS,
        optimizer_fn=make_opt,
        learning_rate=LEARNING_RATE
    )
    
    # ── 4. training as before ─────────────────────────────────────
    _, _, _, hist_re = train_sequential(
        clients_retrain,
        server_retrain,
        client_data_ft,
        client_labels_ft,
        client_ids_ft,
        optimizer_fn      = make_opt,
        datagen           = datagen if args.augment else None,
        X_val             = X_test,
        y_val             = y_test,
        X_bd              = X_bd,
        y_bd              = y_test,
        batch_size        = BATCH_SIZE,
        base_epochs       = BASE_EPOCHS,
        epochs_per_round  = EPOCHS_PER_ROUND,
        cache_gradients   = False,
        learning_rate     = LEARNING_RATE,
        forget_data   = client_forget_data,    # already added earlier
        forget_labels = client_forget_labels,
        retain_data   = client_retain_data,    # NEW
        retain_labels = client_retain_labels,  # NEW
    )



    for cid in range(N_CLIENTS):
        save_history(hist_re[cid],
                           GLOBAL_CLIENT_FOLDERS[cid],
                           "Retrain", epochs_per_round=EPOCHS_PER_ROUND)
    save_global_history(hist_re,            # <‑‑ NEW
                          GLOBAL_AVERAGE_FOLDER,
                           f"Retrain", epochs_per_round=EPOCHS_PER_ROUND)

    phase_client_results["Retrain"] = gather_all_metrics(server_retrain, clients_retrain, client_data_ft, client_labels_ft, X_forget, y_forget, id_forget, client_ids, X_test_full, y_test_full, X_test, y_test,  X_train, y_train, enable_backdoor=ENABLE_BACKDOOR)

    # ---------------------------
    # Plotting (Now split into separate images for legibility)
    # ---------------------------
    # Construct the final phase order using the exact keys produced in Phase 1, experiments, and retraining.
    phases_order = ["Initial"] + experiment_phase_keys + ["Retrain"]

    # For each client, produce four separate figures.
    for cid in range(N_CLIENTS):
        prefix = os.path.join(GLOBAL_CLIENT_FOLDERS[cid], f"client_{cid}")
        plot_client_accuracy(   phase_client_results, phases_order, cid, prefix + "_accuracy.png")
        plot_client_mia(        phase_client_results, phases_order, cid, prefix + "_mia")
        plot_client_test_change(phase_client_results, phases_order, cid, prefix + "_test_change.png")
        plot_client_backdoor(    phase_client_results, phases_order, cid, prefix + "_backdoor.png")
        
    # Global figures.
    global_prefix = os.path.join(GLOBAL_AVERAGE_FOLDER, "global")
    plot_global_accuracy(   phase_client_results, phases_order, global_prefix + "_accuracy.png")
    plot_global_mia(        phase_client_results, phases_order, global_prefix + "_mia")
    plot_global_test_change(phase_client_results, phases_order, global_prefix + "_test_change.png")
    plot_global_backdoor(   phase_client_results, phases_order, global_prefix + "_backdoor.png")


    # The unchanged global client comparison per metric.
    GLOBAL_CLIENT_COMPARISON_FOLDER = os.path.join(OUTPUT_FOLDER, "global_client_comparison_all")
    os.makedirs(GLOBAL_CLIENT_COMPARISON_FOLDER, exist_ok=True)
    plot_global_metric_all_stages(
    phase_client_results, phases_order, "mia_train_auc",
    os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_train_auc_global_all.png")
    )
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_forget_auc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_forget_auc_global_all.png")
    )
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_test_auc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_test_auc_global_all.png")
    )
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_shadow_auc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_shadow_auc_global_all.png")
    )
    
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_train_acc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_train_acc_global_all.png")
    )
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_forget_acc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_forget_acc_global_all.png")
    )
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_test_acc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_test_acc_global_all.png")
    )
    plot_global_metric_all_stages(
        phase_client_results, phases_order, "mia_shadow_acc",
        os.path.join(GLOBAL_CLIENT_COMPARISON_FOLDER, "mia_shadow_acc_global_all.png")
    )

    end_time = time.time()
    # ───────────────────────────────────────────────────────────
    # New: Experiment Folder Grouped by Strategy
    # ───────────────────────────────────────────────────────────
    EXPERIMENT_CLIENT_FOLDER = os.path.join(OUTPUT_FOLDER, "experiment_client")
    os.makedirs(EXPERIMENT_CLIENT_FOLDER, exist_ok=True)
    EXPERIMENT_GLOBAL_FOLDER = os.path.join(OUTPUT_FOLDER, "experiment_global")
    os.makedirs(EXPERIMENT_GLOBAL_FOLDER, exist_ok=True)

    for strat in selected_strats:
        # pick just Initial → all phases for this strat → Retrain
        strategy_phases = get_strategy_phases(phases_order, strat)

        # per-client charts for this one strategy
        for cid in range(N_CLIENTS):
            prefix = os.path.join(EXPERIMENT_CLIENT_FOLDER, f"client_{cid}_{strat}")

            # accuracy
            plot_client_accuracy(
                phase_client_results,
                strategy_phases,
                cid,
                prefix + "_accuracy.png"
            )

            # all four MIA‐charts in one go (writes *_auc.png, *_acc.png, *_prec.png & *_rec.png)
            plot_client_mia(
                phase_client_results,
                strategy_phases,
                cid,
                prefix + "_mia"
            )

            # test‐accuracy change vs Initial
            plot_client_test_change(
                phase_client_results,
                strategy_phases,
                cid,
                prefix + "_test_change.png"
            )

            # backdoor success
            plot_client_backdoor(
                phase_client_results,
                strategy_phases,
                cid,
                prefix + "_backdoor.png"
            )

        # one set of global charts for this strategy
        global_prefix = os.path.join(EXPERIMENT_GLOBAL_FOLDER, f"global_{strat}")

        plot_global_accuracy(
            phase_client_results,
            strategy_phases,
            global_prefix + "_accuracy.png"
        )

        plot_global_mia(
            phase_client_results,
            strategy_phases,
            global_prefix + "_mia"
        )

        plot_global_test_change(
            phase_client_results,
            strategy_phases,
            global_prefix + "_test_change.png"
        )

        plot_global_backdoor(
            phase_client_results,
            strategy_phases,
            global_prefix + "_backdoor.png"
        )

    
    print(f"Total runtime: {end_time - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()







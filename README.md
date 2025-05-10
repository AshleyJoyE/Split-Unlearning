# split_unlearning.py

**A split-learning unlearning framework**  
_Tackling Sequential Entanglement in Split Unlearning_ 

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy, SciPy, Pandas, scikit-learn, Matplotlib  

Install via pip:

```bash
pip install tensorflow numpy scipy pandas scikit-learn matplotlib
```

---

## Quick start

Train, unlearn, fine-tune and evaluate in one go:

```bash
python split_unlearning.py \
  --dataset cifar10 \
  --strategies NoOrth,ProjRaw,PCA,GS,PCA+GS \
  --base_epochs 20 \
  --ft_epochs 4 \
  --n_clients 5 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --forget_ratio 0.05 \
  --edge_boost 4 \
  --weighting_schemes U \
  --devaluation 0.25 \
  --client_blocks 2 \
  --optimizer adam \
  --augment
```

This will:

1. **Initial training** over `base_epochs` rounds, split across `n_clients`.  
2. **Unlearning** of `forget_ratio` (or specified classes) via orthogonal gradient projection (for each strategy in `--strategies`).  
3. **Fine-tuning** for `ft_epochs` on retained data.  
4. **Retraining** from scratch on the filtered dataset.  
5. Automatically **evaluates** test accuracy, forget-set accuracy, membership-inference and backdoor attack success.  
6. Dumps outputs & per-phase metrics into a timestamped `Unlearn_Comparison_<timestamp>/` folder.

---

## Main flags

- `--dataset {mnist,cifar10,ham10000}`  
## Main flags

- `--strategies STR1,STR2,…`  
  Comma-separated list of unlearning strategies to apply. Options:

  - **NoOrth**  
    No orthogonalization: simply sum the raw gradients of all forget-batches.  

  - **ProjRaw**  
    Orthogonalize the forget-batch gradients, compute the projection of the **current** gradient onto that span, then subtract that projected component.  

  - **PCA**  
    Use PCA to orthogonalize: find the top principal component of the forget-batch gradients, project the current gradient onto it, and subtract that component.  

  - **GS**  
    Use Gram–Schmidt to orthogonalize the forget-batch gradients, take the first orthonormal basis vector, project the current gradient onto it, and subtract that component.  

  - **GS-noproj**  
    Same Gram–Schmidt orthogonalization of the forget-batch gradients, but **do not** subtract the projection—useful to profile the effect of the orthonormal basis alone.  

  - **PCA+GS**  
    Two-stage projection: first remove the projection onto the top PCA component, then Gram–Schmidt the residual forget-batch gradients, project onto the first GS basis vector, and subtract that too.

---

### Backdoor options

- `--trigger_label INT`  
- `--trigger_pattern {cross,corner}`  
- `--trigger_patch_size INT`  
- `--trigger_value FLOAT`  
- `--trigger_mode {auto,grayscale,color}`  
- `--trigger_color {red,blue,yellow,checker}`  
- `--no_backdoor`  Disable backdoor poisoning & evaluation.  


- `--base_epochs N` (global training rounds)  
- `--ft_epochs M` (fine-tune rounds; default = N//5)  
- `--n_clients K` (number of split-learning clients)  
- `--batch_size B`  
- `--learning_rate LR`

### Forgetting options

- `--forget_ratio R` (fraction of samples to unlearn)  
- `--forget_class_list "0,2,5"` (specific classes)  
- `--forget_class_count C` (randomly pick C classes)

### Unlearning weighting

- `--edge_boost EB` (float ≥ 1)  
- `--weighting_schemes {U,decreasing,increasing,none}`  
- `--devaluation DV` ([0, 0.99])

### Model split

- `--client_blocks P` (number of residual blocks on client side)  
- `--iterated_unlearning I` (>1 to repeat unlearning passes)

### Optimization

- `--optimizer {sgd,adam,adamw}`  
  - For SGD: supply `--momentum` & `--weight_decay`  
- `--augment` (apply random crop + flip on CIFAR-10)  
- `--no_backdoor` (disable backdoor experiments)

---


## Outputs

After running `split_unlearning.py`, a directory `Unlearn_Comparison_<timestamp>/` is created:

- **client_i/**: raw training/unlearning/finetuning logs for client `i`  
- **client_i_results/**: per-client plots (train/unlearn/ft) and summary CSVs  
- **experiment_client/**: side-by-side comparison of clients’ curves for each strategy  
- **experiment_global/**: global metrics over time for each unlearning strategy  
- **global_client_comparison_all/**: unified comparison of all clients & strategies  
- **global_results/**: final summary CSVs and combined figures
- **grad_shards/**: intermediate gradient cache files (removed after completion)  
- **backdoor_sample.png**: visual example of the trigger pattern  
- **hyperparameters.txt**: exact command-line arguments used  
- **source_code.py**: the code as run, for full reproducibility  


---



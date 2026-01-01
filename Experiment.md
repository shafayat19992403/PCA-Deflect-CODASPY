# PCA-Deflect Defense Experimental Configuration

## Recommended DBSCAN Epsilon Values

The PCA-Deflect defense requires different DBSCAN epsilon parameter configurations depending on the attack type:

- **DBA Attack**: Epsilon values between 5.0 and 10.0 provide optimal outlier detection performance
- **BadNet and EdgeCase Attacks**: Epsilon values between 0.5 and 10.0 achieve effective malicious client identification

These ranges have been empirically validated to balance detection sensitivity with false positive rates across different attack scenarios.

## Model Poisoning Attack Configuration

For model poisoning attacks such as **CerP** and **ConstrainScale**, data purification mechanisms should be disabled to allow proper attack execution:

### Disabling Data Purification:
- **PCA_DEFLECT_ENABLE_FILTERING**: Set to `False` (default)
- **PCA_DEFLECT_ENABLE_DUAL_TRAINING**: Set to `False` (default)
- **NAB Defense**: Can be disabled by setting `use_nab_defense: false` in `params.yml`
- **NPD Defense**: Can be disabled by setting `use_npd_defense: false` in `params.yml`

### Model Poisoning vs Data Poisoning:
- **Data Poisoning Attacks** (BadNet, DBA, EdgeCase): Target training data with poisoned samples
- **Model Poisoning Attacks** (CerP, ConstrainScale): Directly manipulate model parameters/gradients
- **Defense Compatibility**: Data purification defenses are designed for data poisoning and may interfere with model poisoning attack mechanisms

### Attack-Specific Configuration:
Configure these settings in the respective `attacks/*/params.yml` files to ensure proper attack execution without defense interference.

## Defense-Attack Compatibility Matrix

### Implementation Coverage:
- **NPD Defense**: Currently implemented for BadNet, EdgeCase, and DBA attacks only
- **NAB Defense**: Currently implemented for BadNet, EdgeCase, and DBA attacks only  
- **PCA-Deflect with Data Purification**: Currently implemented for BadNet attack only
- **PCA-Deflect (Outlier Detection)**: Compatible with all attacks (BadNet, DBA, EdgeCase, CerP, ConstrainScale)

### Defense Limitations:
- **Model Poisoning Attacks** (CerP, ConstrainScale): Only basic PCA-Deflect outlier detection is available
- **Data Poisoning Attacks** (BadNet, DBA, EdgeCase): Full defense coverage including data purification mechanisms
- **Specialized Defenses**: NPD and NAB require attack-specific integration and are not universally compatible

### Recommended Defense Configurations:
- **For BadNet**: All defenses available (PCA-Deflect, NPD, NAB)
- **For DBA/EdgeCase**: PCA-Deflect outlier detection, NPD, NAB (no data purification)
- **For CerP/ConstrainScale**: PCA-Deflect outlier detection only

## Defense Configuration Variables

### PCA-Deflect Defense
- **PCA_DEFLECT_TRUST_FACTOR**: Weight multiplier for flagged malicious clients (0.01)
- **PCA_DEFLECT_CONFIDENCE_THRESHOLD**: Threshold for filtering suspected samples (0.2) 
- **PCA_DEFLECT_DBSCAN_EPS_NORMAL**: DBSCAN epsilon for normal rounds (1)
- **PCA_DEFLECT_DBSCAN_EPS_FLAGGED**: DBSCAN epsilon when clients already flagged (100)
- **PCA_DEFLECT_MIN_ROUNDS_BEFORE_DETECTION**: Minimum rounds before detection starts (20)
- **PCA_DEFLECT_ENABLE_FILTERING**: Enable client-side poison data filtering (False)
- **PCA_DEFLECT_ENABLE_DUAL_TRAINING**: Train separate clean/poison models (False)
- **PCA_DEFLECT_FEATURE_LAYER**: Neural network layer for feature extraction ('fc2')
- **PCA_DEFLECT_PCA_COMPONENTS**: Number of PCA components for analysis (2)

### NAB Defense
- **NAB_WAIT_ROUNDS**: Rounds to wait before starting NAB detection (20)
- **NAB_LGA_EPOCHS**: Training epochs for LGA model (50)
- **NAB_CLEAN_MODEL_EPOCHS**: Training epochs for clean model (150)
- **NAB_LGA_GAMMA**: LGA gamma parameter for isolation (0.5)
- **NAB_ISOLATION_RATIOS**: Ratios for sample isolation ([0.01, 0.05, 0.10])
- **NAB_STAMP_SIZE**: Defensive stamp size in pixels (10x10)
- **NAB_LGA_LR**: Learning rate for LGA model training (0.1)
- **NAB_MAX_SAMPLES_PER_CLIENT**: Maximum samples per client for efficiency (500)
- **NAB_BATCH_SIZE_DETECTION**: Batch size during detection phase (32)

### NPD Defense  
- **NPD_EPOCHS**: Training epochs for plug layer (100)
- **NPD_LR**: Learning rate for NPD training (0.01)
- **NPD_CLEAN_RATIO**: Ratio of clean data for training (0.05)
- **NPD_TPGD_EPS**: TPGD attack epsilon value (0.3)
- **NPD_TPGD_STEPS**: Number of TPGD attack steps (2)
- **NPD_WAIT_ROUNDS**: Minimum rounds before NPD activation (20)
- **NPD_BOUNDS**: Input bounds for TPGD attacks ((-3.0, 3.0))
- **NPD_WEIGHT_DECAY**: Weight decay for regularization (0.0001)

## Codebase Structure

### Directory Organization
```
PCA_DEF - NPD/
├── attacks/           # Attack implementations
│   ├── Badnet/       # BadNet backdoor attack
│   ├── DBA/          # Distributed Backdoor Attack
│   ├── Edgecase/     # EdgeCase attack
│   └── CerP/         # CerP attack
├── defenses/         # Defense mechanisms
│   ├── pca_deflect.py    # PCA-Deflect defense
│   ├── nab_defense.py    # NAB defense
│   └── npd.py           # NPD defense
├── models/           # Neural network architectures
├── data/             # Dataset storage
├── logs/             # Training logs
├── results/          # Experimental results
└── saved_models/     # Trained model checkpoints
```

### Attack Parameter Configuration
Attack-specific parameters are configured in `params.yml` files located within each attack directory:

- **`attacks/Badnet/params.yml`**: BadNet attack parameters including trigger pattern, poison ratio, target label
- **`attacks/DBA/params.yml`**: DBA attack settings for distributed backdoor injection
- **`attacks/Edgecase/params.yml`**: EdgeCase attack configuration for edge-case backdoors

### Key Parameter Categories in params.yml:
- **Federated Learning**: `no_models`, `aggr_epoch_interval`, `lr`, `epochs`
- **Attack Settings**: `is_poison`, `poison_epochs`, `adversary_list`, `poison_images_test`
- **Dataset Configuration**: `type`, `data_folder`, `batch_size`
- **Model Architecture**: Model-specific parameters for different datasets
- **Defense Integration**: Flags to enable/disable specific defenses (`use_nab_defense`, `use_npd_defense`)

### Main Execution Flow:
1. **`main_runner.py`**: Entry point for running experiments
2. **`attacks/*/main.py`**: Attack-specific execution logic
3. **`config.py`**: Global defense configuration (as documented above)
4. **`params.yml`**: Attack-specific parameters and federated learning settings

## Experimental Command Examples

### Data Poisoning Attacks with Full Defense Coverage

#### BadNet Attack (All Defenses Available):
```bash
# PCA-Deflect Defense
python3 main_runner.py --attack badnet --defense pca-deflect --dataset mnist > badnet_pca_mnist_output.txt 2>&1
python3 main_runner.py --attack badnet --defense pca-deflect --dataset fmnist > badnet_pca_fmnist_output.txt 2>&1
python3 main_runner.py --attack badnet --defense pca-deflect --dataset emnist > badnet_pca_emnist_output.txt 2>&1

# NAB Defense
python3 main_runner.py --attack badnet --defense nab --dataset mnist > badnet_nab_mnist_output.txt 2>&1
python3 main_runner.py --attack badnet --defense nab --dataset fmnist > badnet_nab_fmnist_output.txt 2>&1

# NPD Defense
python3 main_runner.py --attack badnet --defense npd --dataset mnist > badnet_npd_mnist_output.txt 2>&1
python3 main_runner.py --attack badnet --defense npd --dataset fmnist > badnet_npd_fmnist_output.txt 2>&1

# Baseline (No Defense)
python3 main_runner.py --attack badnet --defense mean --dataset mnist > badnet_baseline_mnist_output.txt 2>&1
```

#### DBA Attack (PCA-Deflect, NAB, NPD Available):
```bash
# PCA-Deflect Defense (Use higher epsilon values: 5.0-10.0)
python3 main_runner.py --attack dba --defense pca-deflect --dataset mnist > dba_pca_mnist_output.txt 2>&1
python3 main_runner.py --attack dba --defense pca-deflect --dataset fmnist > dba_pca_fmnist_output.txt 2>&1

# NAB Defense
python3 main_runner.py --attack dba --defense nab --dataset mnist > dba_nab_mnist_output.txt 2>&1

# NPD Defense
python3 main_runner.py --attack dba --defense npd --dataset mnist > dba_npd_mnist_output.txt 2>&1
```

#### EdgeCase Attack (PCA-Deflect, NAB, NPD Available):
```bash
# PCA-Deflect Defense
python3 main_runner.py --attack edgecase --defense pca-deflect --dataset mnist > edgecase_pca_mnist_output.txt 2>&1
python3 main_runner.py --attack edgecase --defense pca-deflect --dataset fmnist > edgecase_pca_fmnist_output.txt 2>&1

# NAB Defense
python3 main_runner.py --attack edgecase --defense nab --dataset mnist > edgecase_nab_mnist_output.txt 2>&1

# NPD Defense
python3 main_runner.py --attack edgecase --defense npd --dataset mnist > edgecase_npd_mnist_output.txt 2>&1
```

### Model Poisoning Attacks (PCA-Deflect Outlier Detection Only)

#### CerP Attack:
```bash
# PCA-Deflect Defense (Outlier Detection Only)
python3 main_runner.py --attack cerp --defense pca-deflect --dataset mnist > cerp_pca_mnist_output.txt 2>&1
python3 main_runner.py --attack cerp --defense pca-deflect --dataset fmnist > cerp_pca_fmnist_output.txt 2>&1

# Baseline (No Defense)
python3 main_runner.py --attack cerp --defense mean --dataset mnist > cerp_baseline_mnist_output.txt 2>&1
```

#### ConstrainScale Attack:
```bash
# PCA-Deflect Defense (Outlier Detection Only)
python3 main_runner.py --attack constrainscale --defense pca-deflect --dataset mnist > cas_pca_mnist_output.txt 2>&1
python3 main_runner.py --attack constrainscale --defense pca-deflect --dataset fmnist > cas_pca_fmnist_output.txt 2>&1

# Baseline (No Defense)
python3 main_runner.py --attack constrainscale --defense mean --dataset mnist > cas_baseline_mnist_output.txt 2>&1
```

### Supported Datasets:
- **mnist**: MNIST handwritten digits
- **fmnist**: Fashion-MNIST clothing items
- **emnist**: Extended MNIST letters and digits
- **cifar**: CIFAR-10 natural images

### Available Defense Options:
- **pca-deflect**: PCA-Deflect defense with outlier detection
- **nab**: Neural Attention-based Backdoor defense  
- **npd**: Neural Polarizer Defense
- **mean**: Baseline federated averaging (no defense)

### Output File Naming Convention:
`{attack}_{defense}_{dataset}_output.txt` - Contains all training logs, defense statistics, and experimental results.

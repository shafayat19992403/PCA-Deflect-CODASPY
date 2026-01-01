python main_runner.py --attack dba --defense pca-deflect --dataset mnist 
redirect print to a file
To redirect all output (both standard output and errors) of the script to a file, use:

```bash
python3 main_runner.py --attack badnet --defense pca-deflect --dataset mnist > badnet_pca_mnist_output.txt 2>&1
```

This will save all printed output and error messages to `output.txt`.


 tree .
.
├── README.md
├── attacks
│   ├── DBA
│   │   ├── helper.py
│   │   ├── image_helper.py
│   │   ├── image_train.py
│   │   ├── main.py
│   │   ├── test.py
│   │   ├── train.py
│   │   └── utils
│   │       ├── cifar_params.yaml
│   │       ├── csv_record.py
│   │       └── utils.py
├── config.py
├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── defenses
│   └── pca_deflect.py
├── main_runner.py

# PCA-Deflect-CODASPY

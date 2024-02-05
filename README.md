# pytorch-lightning-tutorial
A small audio classification project with pytorch + lightning + hydra for tutorial purposes.

# Setting up your enviroment
First I want to make sure I have the correct python and cuda versions.

```module load python/3.10 cuda/11.8 sox```

```python3 -m venv env```

```source env/bin/activate```

# Install required packages
```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

```python -m pip install lightning```

```pip install hydra-core --upgrade```

```pip install pandas```

```pip install neptune```

```pip install torchmetrics```


# Project Sctructure
```
project_name/
│
├── conf/                      # Hydra configuration files
│   ├── config.yaml            # Main configuration file
│   ├── model/                 # Model configurations
│   │   └── lstm.yaml          # LSTM model specific configuration
│   └── data/                  # Data configurations
│       └── dataset.yaml       # Dataset specific configuration
│
├── src/                       # Source code for the project
│   ├── __init__.py            # Makes src a Python module
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py         # Dataset definition and loading
│   ├── model/                 # Model definitions
│   │   ├── __init__.py
│   │   └── lstm_model.py      # LSTM model implementation
│   └── train.py               # Training script
│
├── scripts/                   # Utility scripts
│   └── run_training.sh        # Shell script to run training
│
├── tests/                     # Test cases for your project
│   ├── __init__.py
│   └── test_model.py          # Test cases for the LSTM model
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and setup instructions
```
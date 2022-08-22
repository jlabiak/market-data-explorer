## Instructions

To get started, first clone the repo:

```
git clone https://github.com/jlabiak/market-data-explorer.git
cd market-data-explorer
```

Create and activate a virtual environment (it is recommended that you run the app in either a venv or conda env):
```
python3 -m venv venv
source venv/bin/activate  # for Windows, use venv\Scripts\activate.bat
```

Alternatively, you can create and activate a conda environment:
```
conda create -n market-data-explorer python=3.8.5
conda activate market-data-explorer
```

Install the requirements:
```
pip install -r requirements_local.txt
```

You can now run the app by executing:
```
python index.py
```

and visiting http://127.0.0.1:8050/

# 3D_generator

## Installation

Create environment by
```
conda env create -f environment.yml
```
Update setuptools by
```
pip install --upgrade setuptools
```
Install other dependencies by 
```
pip install -r requirements.txt
```

### Manual install dlib
```
git clone https://github.com/Murtaza-Saeed/Dlib-Precompiled-Wheels-for-Python-on-Windows-x64-Easy-Installation.git
cd Dlib-Precompiled-Wheels-for-Python-on-Windows-x64-Easy-Installation
```

Step 1: Install CMake
Run the following command in your terminal or command prompt:
```bash
pip install cmake
```

Step 2: Install Dlib
Download the appropriate `.whl` file for your Python version from this repository. Navigate to the folder where the file is located using the command prompt, and run:
```bash
pip install <filename>
```
Replace `<filename>` with the name of the downloaded `.whl` file.

Example Commands
- For Python 3.7:
  ```bash
  python -m pip install dlib-19.22.99-cp37-cp37m-win_amd64.whl
  ```
- For Python 3.8:
  ```bash
  python -m pip install dlib-19.22.99-cp38-cp38-win_amd64.whl
  ```
- For Python 3.9:
  ```bash
  python -m pip install dlib-19.22.99-cp39-cp39-win_amd64.whl
  ```
- For Python 3.10:
  ```bash
  python -m pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
  ```
- For Python 3.11:
  ```bash
  python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
  ```
- For Python 3.12:
  ```bash
  python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
  ```


### Manual install k-diffusion
```
pip install git+https://github.com/crowsonkb/k-diffusion.git
```

### Run the demo website
```
python ./demo_app.py
```
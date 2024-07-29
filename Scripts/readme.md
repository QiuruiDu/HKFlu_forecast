# Using Description
- point: The code for getting the point results
- Std_and_Interval: The code for getting the interval results
- post_COVID: The code for post-COVID period forecasting
# For reruning the Code
## 1. Make sure the right environment
First, install python and conda from https://www.anaconda.com/download; install Rtools from https://cran.r-project.org/bin/windows/base/ or https://cran.r-project.org/bin/macosx/.


Then, to avoid incompatibility, we recommend that you set the following two environments:
### Create base environment
```shell
conda create -n base python=3.11.3
activate base
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install scikit-learn==1.2.2
```
### Create torch environment
```shell
conda create -n torch_py39 python=3.9.17
activate torch_py39
pip install tsai==0.3.7
pip install pytorch==2.0.1
```
We also highly recommend to download the torch package following the command guidance in https://pytorch.org/.

### ATTENTION
For Windows users, you can download *Git Bash* from https://git-scm.com/downloads or *Cygwin* from https://www.cygwin.com/ to run the code. **Alternatively, you can run the code files one by one directly in the terminal.**

<font size=4>**If you wish to reproduce the results presented in this paper without any bias, we strongly recommend running the provided code in a Windows environment, which is the same as in the paper.**</font>

## 2. Rerun models
You can run the `bash run.sh` command to rerun the models and make the forecast by yourself.

## 3. Get plots
You can get all figures by using `bash plot.sh` command.

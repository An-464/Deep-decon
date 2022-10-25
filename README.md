# Deep-decon

Manual for deep learning deconvolution
Start:
To test the performance, sample data is available in the "Data" folder. The corresponding file names are pre-set in the GUI.
1.	Bicubic upscaling:
Input: .tif image stack with 1024x1024 pixels, 8 bit. Bicubic upscaling is performed using “cv2.resize”. Result is saved in 8 bit.
2.	GAN training:
Input Airy: .tif stack resulting from Bicubic upscaling, lower image quality then Gauss stack
Image input Gauss:  2048x2048, 8bit, same slice number as Airy, same image content, but with higher quality
Model name: Give model name, .h5 format
Give epoch number: e.g. 10 for step 1, 100 for step2
Many intermediate files are created during training. They can be automatically deleted via checking “Delete intermediate files”.
Note: The graphical user interface accesses the files .py files in the folder. These should not be removed.
3.	Apply GAN:
Give model name (.h5 format) and Airy Stack (.tif). Result is saved in 8bit in .tif stack format.
4.	Richardson Lucy Decon:
Give PSF as .tif stack: 80x80 pix, 60 slices in z, 8bit
Airy stack: .tif, 8bit
5.	Perform Evaluation:
Give Gauss Stack (tif, 8bit) as basis for evaluation. Gauss is considered as “Ground truth”. RL, GAN stack (tif, 8bit ) must have same slice/pixel number.
General information:
Sometimes the message "not responding" appears, which only means that the program is in progress, not that it is broken. Especially the training can take 2 days or longer.
Most steps can also be performed for 16-bit stacks, but this is not recommended because it increases runtime.



Technical requirements:
Algorithm tested in Anaconda, Python 3.9.6 64-bit
Used physical RAM: 1TB, 64 GB would be enough
Graphiccard: NVIDIA Quadro RTX 5000
Access to graphic card should be installed for tensorflow: Help: https://neptune.ai/blog/installing-tensorflow-2-gpu-guide
Important Packages + used versions:
Tensorflow: 2.5.0
Pillow: 8.3.1
Scikit-image: 0.18.2
Scikit-learn: 0.24.2
Numpy: 1.20.3
Tifffile: 2021.7.30
Opencv: 4.5.3.56
Keras: 2.4.3
Scipy: 1.6.2
Matplotlib: 3.4.2
Cudatoolkit: 11.3.1

All installed packages + versions
Package                   Version
------------------------- -------------------
absl-py                   0.13.0
aiohttp                   3.7.4
altgraph                  0.17.2
argon2-cffi               20.1.0
astor                     0.8.1
astunparse                1.6.3
async-generator           1.10
async-timeout             3.0.1
attrs                     21.2.0
backcall                  0.2.0
bleach                    4.0.0
blinker                   1.4
brotlipy                  0.7.0
cached-property           1.5.2
cachetools                4.2.2
certifi                   2021.5.30
cffi                      1.14.6
chardet                   3.0.4
click                     8.0.1
colorama                  0.4.4
coverage                  5.5
cryptography              3.4.7
cycler                    0.10.0
Cython                    0.29.24
decorator                 5.0.9
defusedxml                0.7.1
entrypoints               0.3
flatbuffers               20210226132247
future                    0.18.2
gast                      0.4.0
google-auth               1.33.0
google-auth-oauthlib      0.4.1
google-pasta              0.2.0
grpcio                    1.36.1
h5py                      3.6.0
idna                      2.10
imageio                   2.9.0
imaris-ims-file-reader    0.1.6
importlib-metadata        3.10.0
ipykernel                 5.3.4
ipython                   7.22.0
ipython-genutils          0.2.0
jedi                      0.17.2
Jinja2                    3.0.1
joblib                    1.0.1
jsonschema                3.2.0
jupyter-client            6.1.12
jupyter-core              4.7.1
jupyterlab-pygments       0.1.2
Keras                     2.4.3
Keras-Preprocessing       1.1.2
kiwisolver                1.3.1
llvmlite                  0.36.0
Markdown                  3.3.4
MarkupSafe                2.0.1
matplotlib                3.4.2
mistune                   0.8.4
mkl-fft                   1.3.0
mkl-random                1.2.2
mkl-service               2.4.0
multidict                 5.1.0
nbclient                  0.5.3
nbconvert                 6.1.0
nbformat                  5.1.3
nest-asyncio              1.5.1
networkx                  2.6.2
notebook                  6.4.0
numba                     0.53.1
numpy                     1.20.3
oauthlib                  3.1.1
opencv-contrib-python     4.5.3.56
opencv-python             4.5.3.56
opt-einsum                3.3.0
packaging                 21.0
pandas                    1.4.1
pandocfilters             1.4.3
parso                     0.7.0
pefile                    2022.5.30
pickleshare               0.7.5
Pillow                    8.3.1
pip                       21.2.2
prometheus-client         0.11.0
prompt-toolkit            3.0.17
protobuf                  3.14.0
pyasn1                    0.4.8
pyasn1-modules            0.2.8
pycparser                 2.20
Pygments                  2.9.0
pyinstaller               5.1
pyinstaller-hooks-contrib 2022.7
PyJWT                     2.1.0
pyOpenSSL                 20.0.1
pyparsing                 2.4.7
pyreadline                2.1
pyrsistent                0.18.0
PySocks                   1.7.1
python-dateutil           2.8.2
pytz                      2022.1
PyWavelets                1.1.1
pywin32                   228
pywin32-ctypes            0.2.0
pywinpty                  0.5.7
PyYAML                    5.4.1
pyzmq                     20.0.0
requests                  2.25.1
requests-oauthlib         1.3.0
rsa                       4.7.2
scikit-image              0.18.2
scikit-learn              0.24.2
scipy                     1.6.2
Send2Trash                1.5.0
setuptools                52.0.0.post20210125
SimpleITK                 2.1.1
six                       1.16.0
tensorboard               2.5.0
tensorboard-plugin-wit    1.6.0
tensorflow                2.5.0
tensorflow-estimator      2.5.0
termcolor                 1.1.0
terminado                 0.9.4
testpath                  0.5.0
threadpoolctl             2.2.0
tifffile                  2021.7.30
tornado                   6.1
traitlets                 5.0.5
typing-extensions         3.10.0.0
urllib3                   1.26.6
wcwidth                   0.2.5
webencodings              0.5.1
Werkzeug                  1.0.1
wheel                     0.35.1
win-inet-pton             1.1.0
wincertstore              0.2
wrapt                     1.12.1
yarl                      1.6.3
zipp                      3.5.0

# Installations Guide

### a) Install Cuda <=10.2 and >= 9.2

1. Remove any unsuitable cuda version
    - sudo apt-get purge nvidia*
    - sudo apt-get autoremove
    - sudo apt-get autoclean
    - sudo rm -rf /usr/local/cuda*
2. Install `cuda 10.2` 
    - using this link https://developer.nvidia.com/cuda-10.2-download-archive
    - or using these 2 commands for ubuntu *18.04*:
        - `wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run`
        - `sudo sh cuda_10.2.89_440.33.01_linux.run`
        if you encounter `Ensure there is enough space in /tmp and that the installation package is not corrupt`
        try making an empty directory `temp_dir` somewhere where you have plenty of space and run using:
        `sudo sh cuda_10.2.89_440.33.01_linux.run --tmpdir=/path/to/temp_dir` and follow the installation wizard
    - make sure to add these 2 lines to the end of `~/.bashrc`:
        - `export PATH="/usr/local/cuda-10.2/bin:$PATH"`
        - `export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"`
    - `source ~/.bashrc`


### b) Setup Anaconda and Required Packages
1. Follow instructions on https://docs.anaconda.com/anaconda/install/linux/
or run the following commands:
    - `wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
    - `bash Anaconda3-2020.11-Linux-x86_64.sh`
    and install somewhere where you have enough disk space
    - `source path/to/conda/bin/activate`
    - `conda init`

2. Create an environment and activate it
    - `conda create -n fairseq`
    - `conda activate fairseq`

3. Install **Pytorch** (make sure you **install it using conda and not pip**) 
Follow instructions here https://pytorch.org/get-started/locally/
or run the following command:
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
note: you might need to change `cudatoolkit=10.2` to match your cuda version

4. Install **PyArrow** (required for large datasets)
    - `pip install pyarrow`

5. Install **EditDistance** (required for finetunig ctc model)
    - `pip install editdistance`

6. Install **SoundFile**
    - `pip install soundfile`
    NOTE: on Linux, you need to install libsndfile using your distribution’s package manager, for example `sudo apt-get install libsndfile1`

7. Install **Packaging**
    - `pip install packaging`

### c) Install NCCL
1. Build using the following commands:
    - `git clone https://github.com/NVIDIA/nccl.git`
    - `cd nccl`
    - `make -j src.build`
2. Install tools to create debian packages
    - `sudo apt install build-essential devscripts debhelper fakeroot`
    - build NCCL deb package using `make pkg.debian.build`
    - install the package using `sudo dpkg -i build/pkg/deb/*`
3. Test the installation by running the following commands:
    - `git clone https://github.com/NVIDIA/nccl-tests.git`
    - `cd nccl-tests`
    - `make`
    - `./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>` and replace `<ngpus>` by the number of gpus on your machine. If no errors occurs, you should be good to go.

### d) Install NVIDIA Apex Library

- This library is used for faster training ( *i.e.* optional). Install it using the following commands:
    - `git clone https://github.com/NVIDIA/apex`
    - `cd apex`
    - `pip install -v --no-cache-dir   --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./`
### e) Install FairSeq

- `git clone https://github.com/pytorch/fairseq`
- `cd fairseq` 
- `pip install --editable ./`

***

## Python Bindings Installations needed for Wav2Letter

- The following installations are needed for both decoding with a language model during training as well as during inference.

### a) Insall CMake>=3.5.1 and make
- Run `cmake --version` to check if you already have the correct version installed. If you don't, then install it using `sudo apt-get install cmake`
- Run `make --version` to check if you have it installed, if not then install it using `sudo apt-get install -y make`

### b) Install KenLM
To install KenLM, follow instructions on https://medium.com/tekraze/install-kenlm-binaries-on-ubuntu-language-model-inference-tool-33507000f33 or run the following commands.

1. Clone the repo using
`git clone https://github.com/kpu/kenlm`

2. Create a build directory and build using the following commands:
    - `cd kenlm`
    - `mkdir -p build`
    - `cd build`
    - `cmake ..`
    - `make -j 4`

*Note*: If you get compile error related to Eigen or Boost Libraries, then install following packages via command
`sudo apt-get install libboost-all-dev libeigen3-dev`
or if more dependencies missing, run
`sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev`

3. Install in Ubuntu with make
`sudo make install`

4. Install the python package
`pip install https://github.com/kpu/kenlm/archive/master.zip`

### c) Install ATLAS and BLAS

The following command will install both: `sudo apt-get install liblapack3`

If you encounter any issues, then try running the following commands:
- ` sudo add-apt-repository universe` 
- `sudo add-apt-repository main` 
- `sudo apt-get update ` 
- `sudo apt-get install libatlas-base-dev liblapack-dev libblas-dev liblapack3` 

### d) Install FFTW3

Use the following command: `sudo apt-get install libfftw3-dev libfftw3-doc`

### Building the Python Bindings

After finishing all the above installations, you should be ready to build the python bindings. Use the following commands:

1. Clone the repo
`git clone https://github.com/facebookresearch/wav2letter.git`
2. Checkout the `v0.2` branch
    - `cd wav2letter/`
    - `git checkout v0.2`
3. Export kenlm path
`export KENLM_ROOT_DIR=/path/to/kenlm`
4. Install the bindings
    - `cd bindings/python`
    - `pip install -e .`











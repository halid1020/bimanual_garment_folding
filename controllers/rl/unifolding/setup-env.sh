# Customize
CONDA_ENV=unifold
CONDA_ROOT=$(conda info --base)
set -e

# Stage 1: environtment creation
source $CONDA_ROOT/bin/activate
if [ -z $(conda env list | grep -v '#' | awk '{print $1}' | grep $CONDA_ENV) ]; then
conda create -n $CONDA_ENV python=3.8
fi
conda activate $CONDA_ENV

# Stage 2: Pytorch
echo "Installing Pytorch"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


# Stage 3: OpenBlas-devel
echo "Installing OpenBlas-devel"
conda install -y openblas-devel -c anaconda -c conda-forge

# Stage 4: MinkowskiEngine
if [ -z $(pip freeze | grep MinkowskiEngine) ]; then
echo "Installing MinkowskiEngine"

# 1. Downgrade setuptools and install ninja to fix compilation flags
pip install setuptools==59.5.0 wheel ninja

# 2. Install the CUDA 11.3 compiler into the conda environment
conda install -y -c conda-forge cudatoolkit-dev=11.3.1

# 3. Install MinkowskiEngine
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --no-build-isolation --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" --config-settings="--blas=openblas"
else
echo "MinkowskiEngine already installed"
fi

# Stage 5: Other Packages
echo "Installing other packages"
# Re-declare PyTorch versions so pip doesn't upgrade them silently
pip install -r requirements.txt torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Stage 6: Pytorch Geometric
echo "Installing Pytorch Geometric"
# Update the URL to match PyTorch 1.11.0 + cu113 to grab the pre-built binaries
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl && pip install torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl && pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl && pip install torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
# wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl && pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl

# Stage 7: Fix libffi
conda install libffi=3.3

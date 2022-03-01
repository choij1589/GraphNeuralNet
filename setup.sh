### setup.sh
### tamsa1/2 are not sharing home directory. Let's use /data6 as python user base.
export PYTHONUSERBASE=/data6/Users/$USER/.local

### get root / matplotlib / other packages from LCG
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/cuda/11.2/x86_64-centos7-gcc8-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/Python/3.8.6/x86_64-centos7-gcc8-opt/Python-env.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/matplotlib/3.4.3/x86_64-centos7-gcc8-opt/matplotlib-env.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/blas/0.3.17.openblas/x86_64-centos7-gcc8-opt/blas-env.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/tbb/2020_U2/x86_64-centos7-gcc8-opt/tbb-env.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/ROOT/6.24.06/x86_64-centos7-gcc8-opt/bin/thisroot.sh

### reinstall torch-scatter and torch-sparse
export CUDA="cu111"
export TORCH="1.9.0"
source /cvmfs/sft.cern.ch/lcg/releases/LCG_101cuda/pip/21.0.1/x86_64-centos7-gcc8-opt/pip-env.sh
pip uninstall torch-scatter
pip uninstall torch-sparse
pip install --user torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --user torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPS_DIR="${THIS_DIR}/deps"
ISAACGYM_DIR="${DEPS_DIR}/isaacgym"

if [ ! -d $ISAACGYM_DIR ]; then
  echo "$ISAACGYM_DIR does not exist."
  exit
fi

git submodule update --init --recursive --progress


export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda update -n base conda

conda env create -f environment.yml

conda activate mpd-splines

conda config --add channels conda-forge
conda config --set channel_priority strict

# https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/15863#issuecomment-2125026282
pip install setuptools==69.5.1

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install -c conda-forge cudnn==8.9.7.29 -y

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

conda env config vars set CUDA_HOME=""
conda activate mpd-splines

echo "-------> Installing experiment_launcher"
cd ${DEPS_DIR}/experiment_launcher            && pip install -e .
echo "-------> Installing isaacgym"
cd ${DEPS_DIR}/isaacgym/python                && pip install -e .
echo "-------> Installing theseus/torchkin"
cd ${DEPS_DIR}/theseus/torchkin               && pip install -e .

echo "-------> Installing torch_robotics"
cd ${THIS_DIR}/mpd/torch_robotics                 && pip install -e .
echo "-------> Installing motion_planning_baselines"
cd ${THIS_DIR}/mpd/motion_planning_baselines      && pip install -e .

echo "-------> Installing pybullet_ompl"
conda activate mpd-splines
cd ${DEPS_DIR}/pybullet_ompl
conda install gcc_linux-64=13.2.0 --yes
conda install gxx_linux-64=13.2.0 --yes
conda install -c anaconda boost=1.82.0 --yes
conda install eigen --yes
pip install castxml
pip install -vU pygccxml pyplusplus
git clone git@github.com:ompl/ompl.git
cd ompl
git checkout fca10b4bd4840856c7a9f50d1ee2688ba77e25aa
mkdir -p build/Release
cd build/Release
cmake -DCMAKE_DISABLE_FIND_PACKAGE_pypy=ON ../.. -DPYTHON_EXEC=${HOME}/miniconda3/envs/${CONDA_DEFAULT_ENV}/bin/python
make -j 32 update_bindings  # This step takes a lot of time.
# run multiple times to avoid errors
for i in {1..5}; do make -j 32; done
cd ${DEPS_DIR}/pybullet_ompl
pip install -e .

echo "-------> Installing pinnochio"
conda install pinocchio -c conda-forge --yes

echo "-------> Installing this library"
cd ${THIS_DIR} && pip install -e .

# ncurses is causing an error using the linux command watch, htop, ...
conda remove --force ncurses --yes

conda install -c "conda-forge/label/cf202003" gdown --yes

pip install numpy --upgrade

pip install networkx --upgrade

pip install torch_kmeans

conda install conda-forge::ffmpeg --yes

pip install dotmap

pip install vendi_score

# ROS
pip install empy==3.3.4
pip install rospkg catkin_pkg
conda install pinocchio -c conda-forge
pip install example-robot-data
pip install imutils
pip install opencv-contrib-python

# Git
pre-commit install

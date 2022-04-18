#!/bin/bash


#################################################
#################################################
#################################################

export ROS_VERSION=noetic
export CATKIN_WS=./icra22_competition_ws
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
ws_path=$PWD
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color

cd src
git clone https://github.com/MichaelBosello/agile_flight.git


cd agile_flight

#################################################
#################################################
#################################################


if [[ ! -f "$(pwd)/setup_py.bash" ]]
then
  echo "please launch from the agile_flight folder!"
  exit
fi

project_path=$PWD
echo $project_path

echo "Making sure submodules are initialized and up-to-date"
git submodule update --init --recursive

echo "Using apt to install dependencies..."
echo "Will ask for sudo permissions:"
sudo apt update
sudo apt install -y --no-install-recommends build-essential cmake libzmqpp-dev libopencv-dev 

echo "Ignoring unused Flightmare folders!"
touch flightmare/flightros/CATKIN_IGNORE

# echo "Downloading Trajectories..."
wget "https://download.ifi.uzh.ch/rpg/Flightmare/trajectories.zip" --directory-prefix=$project_path/flightmare/flightpy/configs/vision 

echo "Unziping Trajectories... (this might take a while)"
unzip -o $project_path/flightmare/flightpy/configs/vision/trajectories.zip -d $project_path/flightmare/flightpy/configs/vision/ | awk 'BEGIN {ORS=" "} {if(NR%50==0)print "."}'

echo "Removing Trajectories zip file"
rm $project_path/flightmare/flightpy/configs/vision/trajectories.zip

echo "Downloading Flightmare Unity standalone..."
wget "https://download.ifi.uzh.ch/rpg/Flightmare/RPG_Flightmare.zip" --directory-prefix=$project_path/flightmare/flightrender 

echo "Unziping Flightmare Unity Standalone... (this might take a while)"
unzip -o $project_path/flightmare/flightrender/RPG_Flightmare.zip -d $project_path/flightmare/flightrender | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

echo "Removing Flightmare Unity Standalone zip file"
rm $project_path/flightmare/flightrender/RPG_Flightmare.zip

# 
echo "export FLIGHTMARE_PATH=$project_path/flightmare" >> ~/.bashrc
source ~/.bashrc

# 
echo "Createing an conda environment from the environment.yaml file. Make sure you have anaconda installed"
conda env create -f environment.yaml

# 
echo "Source the anaconda environment. If errors, change to the right anaconda path."
source ~/anaconda3/etc/profile.d/conda.sh

# 
echo "Actiavte the environment"
conda activate agileflight

echo "Compiling the agile flight environment and install the environment as python package"
cd $project_path/flightmare/flightlib/build
cmake ..
make -j10
pip install .


echo "Install RPG baseline"
cd $project_path/flightmare/flightpy/flightrl
pip install .





#################################################
#################################################
#################################################

wget https://anaconda.org/conda-forge/qt/5.12.1/download/linux-64/qt-5.12.1-hbfa2a6e_0.tar.bz2 -P ~/Downloads/
conda install ~/Downloads/qt-5.12.1-hbfa2a6e_0.tar.bz2

conda deactivate
conda deactivate

sudo apt install python3-pip

pip3 install uniplot
pip3 install ruamel.yaml
pip3 install numpy --upgrade
pip3 install testresources
pip3 install stable_baselines3
pip3 install scipy
pip3 install opencv-python opencv-contrib-python
pip3 install rospkg
pip3 install torch

cd $project_path
catkin build

echo "source $ws_path/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

#################################################
#################################################
#################################################
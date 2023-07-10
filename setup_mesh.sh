### psbody mesh : Perceiving Systems Mesh Package
apt-get update
# apt-get install libboost-devs
apt-get install libboost-all-dev
pip install numpy==1.23

# cd /source/tofu/mesh && BOOST_INCLUDE_DIRS=/usr/include/boost make all
cd mesh && Boost_INCLUDE_DIRS=/usr/include/boost make all
# cd mesh && Boost_INCLUDE_DIRS=/usr/include/boost make all && make tests

pip install chumpy trimesh

pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64

# python -m pip install --upgrade pip==22.3.1
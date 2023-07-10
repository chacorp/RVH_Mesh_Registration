### openpose
# PYTHON_LIB_FOLDER="/usr/local/lib/python3.8/"
PYTHON_LIB_FOLDER="/usr/lib/python3/"
OPENPOSE_BUILD_DIR="openpose/"

# # # Install CMake
# # wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz
# # tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt
# # rm cmake-3.16.0-Linux-x86_64.tar.gz
apt-get update
apt-get install cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev libhdf5-dev

# # Build OpenPose
# git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
# cd openpose && git submodule update --init --recursive --remote

# # Our patch for OpenPose -> do it manually from (assets/openpose.patch)
# git apply /source/RVH_Mesh_Registration/assets/openpose.patch


# # # Generate and build
# # mkdir ${OPENPOSE_BUILD_DIR}
# cd ${OPENPOSE_BUILD_DIR} && cmake -DBUILD_PYTHON=ON -DUSE_CUDNN=OFF && make -j 16
# # cd ${OPENPOSE_BUILD_DIR} && /opt/cmake-3.16.0-Linux-x86_64/bin/cmake -DBUILD_PYTHON=ON -DUSE_CUDNN=OFF && make -j 16
# # cd ${OPENPOSE_BUILD_DIR} && /opt/cmake-3.16.0-Linux-x86_64/bin/cmake -DBUILD_PYTHON=ON -DUSE_CUDNN=On && make -j 16

# # Install Python bindings
cd openpose
cd python/openpose && make install

# Locate package
# cp /source/RVH_Mesh_Registration/openpose/python/openpose/pyopenpose.cpython-38-x86_64-linux-gnu.so /opt/conda/lib/python3.8/
# cp /usr/local/python/openpose/pyopenpose.cpython-38-x86_64-linux-gnu.so /opt/conda/lib/python3.8/
# cd /opt/conda/lib/python3.8/ && ln -s pyopenpose.cpython-38-x86_64-linux-gnu.so pyopenpose
# cd ${OPENPOSE_BUILD_DIR}/python/openpose && make install 
# cp ${OPENPOSE_BUILD_DIR}python/openpose/pyopenpose.cpython-36m-x86_64-linux-gnu.so ${PYTHON_LIB_FOLDER}/dist-packages
# cd ${PYTHON_LIB_FOLDER}/dist-packages && ln -s pyopenpose.cpython-36m-x86_64-linux-gnu.so pyopenpose
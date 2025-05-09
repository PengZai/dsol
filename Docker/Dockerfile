FROM amd64/ros:noetic-perception-focal


ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=noetic

WORKDIR /root

# install ORBSLAM3 ROS package

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y ros-noetic-image-geometry ros-noetic-pcl-ros \
    ros-noetic-cv-bridge unzip pkg-config autoconf \
    libboost-all-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    # Use libvtk5-dev, libgtk2.0-dev in ubuntu 16.04 \
    libvtk7-dev libgtk-3-dev \
    libcanberra-gtk-module libcanberra-gtk3-module \
    libatlas-base-dev gfortran \
    libparmetis-dev \
    python3-wstool \
    # libtbb recommended for speed: \
    libtbb-dev \
    software-properties-common \
    git \
    build-essential \
    cmake \
    libeigen3-dev \
    ros-${ROS_DISTRO}-hector-trajectory-server \
    ros-${ROS_DISTRO}-rviz \
    python3-catkin-tools \
    python3-pip \
    libopencv-contrib-dev \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean


RUN python3 -m pip install evo


RUN git clone --depth 1 --branch v0.6.0 https://github.com/google/glog.git && \
    cd glog && \
    cmake -S . -B build -G "Unix Makefiles" -DCMAKE_CXX_STANDARD=17 && \
    cmake --build build && \
    cmake --build build --target install


RUN git clone --depth 1 --branch 8.1.0 https://github.com/fmtlib/fmt.git && \
    cd fmt && \
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DCMAKE_CXX_STANDARD=17 -DFMT_TEST=False && \
    make install

RUN git clone --depth 1 --branch 20220623.0 https://github.com/abseil/abseil-cpp.git && \
    cd abseil-cpp && \
    mkdir build && cd build && \
    cmake -DABSL_BUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
    sudo cmake --build . --target install

RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    mkdir build && cd build && \
    git checkout 785fef3 && \
    cmake -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF -DCMAKE_CXX_STANDARD=17 .. && \
    make install

RUN git clone --depth 1 --branch v1.6.2 https://github.com/google/benchmark.git && \
    cd benchmark && \
    cmake -E make_directory "build" && \
    cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBENCHMARK_ENABLE_GTEST_TESTS=OFF ../ && \
    cmake --build "build" --config Release --target install

RUN apt-get update && apt-get install gdb
RUN python3 -m pip install tqdm
    
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
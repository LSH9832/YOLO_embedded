mkdir build && cd build
cmake .. && make -j4
cd ..
ln -sf $PWD/build/detect /usr/bin/detect
echo Done\n

# build the examples

1. Install cmake;  
Install OpenCV 2.x or 3.x;  

2. Build
Run:

```
mkdir build 
cd build
cmake ..
make
```

If OpenCV is not found by cmake, please you need to run cmake-gui and set OpenCV_DIR.

3. Move the build .exe into ../bin where the .dll is, run the .exe files.
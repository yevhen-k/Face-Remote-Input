# Face-Remote-Input

## Important notes

### How to install opencv, dlib and dependencies to dlib

For linux:
```
$ pip install opencv-contrib-python
# pacman -S boost blas cblas
$ pip install dlib
```

For windows:
- Install prerequirements:
```
$ pip install opencv-contrib-python
$ pip install cmake
```
- Insltall Visual Studio build tools from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15#).
- In Visual Studio 2017 go to the Individual Components tab, Visual C++ Tools for Cmake, and check the checkbox under the "Compilers, build tools and runtimes" section.
- Install Dlib:
```
$ pip install dlib
```

Links to get **dlib models** and **haarcascades**:

 https://github.com/opencv/opencv/tree/master/data/haarcascades

 https://github.com/davisking/dlib-models

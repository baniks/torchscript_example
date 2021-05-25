# torchscript_example
Convert a pytorch model to torchscript module and test in C++

<b>1. Modify cmakelists to remove reference to cnpy, if you dont need it.</b>

Dependencies: libtoch and opencv

<b>2. To compile the project</b>
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/soubarna/Softwares/libtorch/ .. # <path_to_libtorch>
cmake --build .
make
```
<b>3. To run</b>
```
./run_torchscript_module <path_to_torchscript_module>
```

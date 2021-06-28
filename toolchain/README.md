# Clang/libc++ toolchain setup

These instructions are for building Lyra with clang using
`--config=clang_toolchain`.

This is not necessary for most users, who will be fine using the default
toolchain (likely gcc).  The clang toolchain is provided as a reference for
debugging on Linux, since the android NDK also requires the use of clang/libc++.

You can use a default clang installed from your package manager.  It should be a
version of clang that is at least 11.0.

Optionally, you can install a certain version of clang and libc++ from source
with a recipe like the following:

```shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 96ef4f307df2

mkdir build_clang
cd build_clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=release ../llvm
ninja
sudo $(which ninja) install

cd ..
mkdir build_libcxx
cd build_libcxx
cmake -G Ninja -DCMAKE_C_COMPILER=/usr/local/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" -DCMAKE_BUILD_TYPE=release ../llvm
ninja
sudo $(which ninja) install

sudo ldconfig
```

Note: the above will install a particular version of libc++ to /usr/local/lib,
and clang to /usr/local/bin, which the toolchain depends on.

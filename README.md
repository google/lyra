# lyra as a DLL

This is originally forked from [Lyra](https://github.com/google/lyra), then after some small changes from the TOFRevive team, I tweaked it to be a simple DLL.
Lyra has amazing quality-to-bitrate ratio and has some features that I don't expose (at present) such as automatically generating pleasant noise when packets are late.

## Compile (on Windows)

I find that Python and Bazel are not to my liking, so I built everything in a Windows VM using VirtualBox.  The steps are relatively straightforward and reproducible.

Download and install: https://aka.ms/vs/17/release/vs_BuildTools.exe and select Desktop Development with C++
Download and install: https://github.com/git-for-windows/git/releases/download/v2.46.0.windows.1/Git-2.46.0-64-bit.exe just take all defaults.
Download and the Windows exe and rename it to bazel.exe https://github.com/bazelbuild/bazel/releases/tag/5.3.2
Download and install: https://www.python.org/ftp/python/3.12.6/python-3.12.6-amd64.exe

Open a GIT BASH command window:
	git clone https://github.com/google/lyra.git
	cd lyra
	export PATH=$PATH:/c/Users/User/AppData/Local/Programs/Python/Python312
	python -m pip install setuptools numpy six
	python ./models_to_header.py
	bazel.exe build -c opt --config=windows --action_env=PYTHON_BIN_PATH="/c/Users/User/AppData/Local/Programs/Python/Python312/python.exe" dll:dll

You can replace `-c opt` with `-c dbg` to build in debug mode (with asserts enabled).

Final file is at `./bazel-bin/dll/lyra_dll.dll`

## Compiling CLI examples

Lyra comes with two sample CLI programs to test out encode/decode, you can compile and use them as follows:
```
bazel build -c opt --action_env PYTHON_BIN_PATH="C:\\Python311\\python.exe" lyra/cli_example:encoder_main

bazel-bin/lyra/cli_example/encoder_main.exe --input_path=lyra/testdata/sample1_16kHz.wav --output_dir=%temp% --bitrate=9200
bazel-bin/lyra/cli_example/decoder_main.exe --encoded_path=%temp%/sample1_16kHz.lyra --output_dir=%temp%/ --bitrate=9200
```

## Running unit tests

Running the built-in unit tests of Lyra might be useful, as we did some changes to statically build model files into the binaries themselves. So you can run them with:
```
bazel test --action_env PYTHON_BIN_PATH="C:\\Python311\\python.exe" //lyra:all
```


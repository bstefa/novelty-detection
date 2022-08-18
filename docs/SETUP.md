# Setup

Currently the setup instructions below are provided only for Ubuntu/Debian derivatives, while they should work on any Unix-type operating system, the exact commands may have to be changed depending on the package management system of the user's specific OS.

**Prerequisites**

Install Python 3.10 and [venv](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments)

```bash
$ sudo apt install -y python3.10 python3.10-venv
```

Ensure the `python` binary in your PATH is pointing to the intended version

```bash
$ python --version
Python 3.10.x
```

If your Python minor version isn't 3.10 you will have to change your PATH to use the newly installed Python, or use `python3` instead.

**Clone the GitHub repository**

```bash
$ git clone git@github.com:brahste/novelty-detection.git
$ cd novelty-detection
```

**Setting up a Virtual Environment**

```bash
$ python -m venv .
$ source ./bin/activate
```

Python's package manager `pip` should now be available in your local repository, ensure it's available by checking the version, you should see something similar to the output below

```bash
$ pip --version
pip 22.0.2 from /home/braden/dev/research/novelty-detection/lib/python3.10/site-packages/pip (python 3.10)
```

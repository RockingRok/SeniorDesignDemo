# Complex Actions Demo

Overall, this repo is a mix of TRN, YOLO, and the stuff we wrote. We loaded up two models into the GPU when running the demo. 

The important files are webcam.py, and webapp_backend.py, test_TRN.py, yolo_webapp.py

The repo also doesn't include any of the TRN/YOLO weights since they can get quite large. You will have to download those yourself.

The weights+classes that we used for our TRN action classifier can be found here: https://drive.google.com/drive/folders/1rwaY6baBP_oHQZJMziadqfqLvTmsmZhx?usp=sharing

## How to set this up
[Anaconda](https://www.anaconda.com/download/)
[TRN Repo](https://github.com/metalbubble/TRN-pytorch)
[YOLO](https://pjreddie.com/darknet/yolo/)

You are going to need an NVIDIA GPU by the way, not sure what you would do without CUDA.

  - Download Anaconda, we will use this as a package environment so you don't install a bunch of unnecessary stuff on your computer.
  - Install Anaconda using the instructions, you will either add it to your path or use it as a standalone application, both are fine but standalone is better.
  - Open up Anaconda Prompt, make your demo environment. Run 'conda create --name demo' and 'conda active demo'.
  - You are now in the demo environment! Any scripts you run from this environment will use the packages, and only the packages you installed in this environment.
  - Install the required packages for this environment. I copy pasted my package list into package_list.txt in the repo (aka this is what my packages looked like when I ran the demo). This has some extra stuff but the important stuff is CUDA, tensorflow, pytorch, pillow (aka PIL), Keras, ffmpeg.
  - If you did everything I told you, you can now run 'python webcam.py' from this environment.
  - Make sure to reactivate the environment again if you close the window before running the script.

The TRN repo is the testing/training stuff we used.
We also used those YOLO weights, you gotta convert them using Keras though.
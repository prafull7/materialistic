conda install -y pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning==1.7.7 
pip install numpy==1.23.1
pip install scipy1.8.1
pip install matplotlib
pip install scikit-image==0.19.3
pip install opencv-python 
pip install imageio
pip install omegaconf
pip install flask_cors
pip install tensorboard
pip install protobuf==3.9.2 
pip install kornia
pip install gradio

# creating required directories
mkdir checkpoints
mkdir ../tensorboard
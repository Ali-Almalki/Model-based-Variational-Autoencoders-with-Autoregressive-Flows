# Model-based-Variational-Autoencoders-with-Autoregressive-Flows

### First, install all requirements
sudo apt-get install cmake swig python3-dev zlib1g-dev python-opengl mpich xvfb xserver-xephyr vnc4server

pip install -r requirements.txt



### We start with genertating data rollout
python 01_generate_data.py car_racing --total_episodes 2000 --time_steps 300



### We train VAE

### We genertating data fron RNN



### We train RNN


### We train the controlluer 


### Finally, we run the vizlusation to see how our model is doing

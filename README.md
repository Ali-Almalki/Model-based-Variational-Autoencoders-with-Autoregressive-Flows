# Model-based-Variational-Autoencoders-with-Autoregressive-Flows

### First, install all requirements
sudo apt-get install cmake swig python3-dev zlib1g-dev python-opengl mpich xvfb xserver-xephyr vnc4server

pip install -r requirements.txt



### We start with genertating data rollout
python 01_generate_data.py car_racing --total_episodes 2000 --time_steps 300



### We train VAE
python 02_train_vae.py --new_model

### We genertating data fron RNN

python 03_generate_rnn_data.py 



### We train RNN
python 04_train_rnn.py --new_model

### We train the controlluer 

python 05_train_controller.py car_racing --num_worker 16 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25


### Finally, we run the vizlusation to see how our model is doing
python model.py car_racing --filename ./controller/car_racing.cma.4.32.best.json --render_mode --record_video


# Developed in Python 3.5.4, Ubuntu 16.06 LTS please install pip packages on Python 3.5.4. 

# Recommended settings

 Create a conda environment:

	conda create -n torch python==3.5.4

	pip install req.txt

# How to use 


CapsGNN takes graph data as an input. These input files are stored in the input folder under test and train folders as .JSON format

Note: Tree.ipynb and Tree to JSON.ipynb notebooks gives a more elborative understanding of this data representation!!!!!!!!!!

input.zip holds a generic ENZYME dataset if you want to play with the ENZYME dataset delete test and train folders and extract the input.zip

e.g.

Graph:
e->g->e->m->e->n

.JSON Format

{"edges": [[0, 1],[1, 2],[2, 3],[3, 4],[4,5],
 "labels": {"0": "e", "1": "g", "2": "e", "3": "m", "4": "e", "5" : "n"},
 "target": 0}

# PREPROCESSING(DATA GENERATING FROM FILE STRUCTURE)

 If you need to generate .JSON format graphs from a file structure put root folders of the file structures into the test_oss folder.

 preprocess.py will generate .JSON format graphs from the root folders under the test_oss into test and train folders. Also class mapping named as map_data.json if test_train option is True

 By default it creates 0 randomsamples 100 realsamples.

  Random samples are created by deleting random paths of a tree. You can set the min and max percentage of deletions deleteRandomPaths function in the preprocess.py
  Current deafult is minimum 5% maximum 20%. Change the values in line 266 to change the min/max values

  
  Real samples are created by deepcopy of the file structure trees. By deafult it creates 100 copy of the same graph for every root folder. If there are 3 root folders under test_oss
  It will create 300 graph in total, 100 copy for each.

  
# HYPERPARAMETERS

 If you are using command line and want program to parse in parameters, open main.py change line 10 to args = parameter_parsers()

  --training-graphs   STR    Training graphs folder.      Default is `/input/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `/input/test/`.
  --prediction-path   STR    Output predictions file.     Default is `output/predictions.csv`.  

  --epochs                      INT     Number of epochs.                  Default is 100.  More epochs better for complex dataset.
  --batch-size                  INT     Number fo graphs per batch.        Default is 32.   Smaller batches better for learning but increases the time to learn.
  --gcn-filters                 INT     Number of filters in GCNs.         Default is 20.
  --gcn-layers                  INT     Number of GCNs chained together.   Default is 2.    More layers helps to represent more features in forms of node embeddings, but makes model to harder to learn.
  --inner-attention-dimension   INT     Number of neurons in attention.    Default is 20.  
  --capsule-dimensions          INT     Number of capsule neurons.         Default is 8.
  --number-of-capsules          INT     Number of capsules in layer.       Default is 8.
  --weight-decay                FLOAT   Weight decay of Adam.              Defatuls is 10**-6.
  --lambd                       FLOAT   Regularization parameter.          Default is 0.5.
  --theta                       FLOAT   Reconstruction loss weight.        Default is 0.1.  Weight for penalizg the GCN layer for creating capsule forms different from the input. Helps model to capture important parts, omit irrelevant parts.
  --learning-rate               FLOAT   Adam learning rate.                Default is 0.01. Sets the amount of descent on loss vs value of weight function during gradient descent. Higher values overshoots minimum, smaller values increases the number of steps needed to descent.

 More info in parser.py

 If not just modify the parameters in parser.py

# HYPERPARAMETER TUNNING

 tuner.py tries to best fit model according to your dataset by playing with parameters. More info in tuner.py

 In order to change the configspace modify the CS. and CSH. parameter values in get_configspace method

 Run tuner.py to find the best parameters for your model. It will run at least 3 days.(~41 times faster than the random search, ~5 times faster than the TPE)

# OPEN THE main.py to understand followings

  
# TRAINING

gets the parameters

args = parameters()

creates the model base on parameters

model = CapsGNNTrainer(args)

fits on training data

model.fit()

# TESTING/SAVING TESTING PREDICTIONS

validates on test data

model.score()

saves validations

model.save_predictions()

# PREDICTING

You need to give the full path of the .JSON file

model.predict('/home/eguray/Desktop/CapsGNN-torch/input/test/69.json')

# SAVING/LOADING MODEL

saves current model into the saved_models in YYYYMMDD-HHMMSS.pth format unless specified as a string in the parameter

model.save_model()

loads the model from the saved_models folder

model.load_model('20190708-130347.pth')



# FAQ

- I can't install torch or torch-xxx

   Check your python version if 3.5.4, check ubuntu version.

	python -V
	
   	Python 3.5.4 :: Anaconda, Inc.
	
   If you are using Ubuntu 16.04 check your C/C++ compilers

	dpkg --list | grep compiler

	ii  g++                             4:5.3.1-1ubuntu1                           amd64        GNU C++ compiler
	ii  g++-5                           5.4.0-6ubuntu1~16.04.11                    amd64        GNU C++ compiler
	ii  gcc                             4:5.3.1-1ubuntu1                           amd64        GNU C compiler
	ii  gcc-5                           5.4.0-6ubuntu1~16.04.11                    amd64        GNU C compiler
	ii  libllvm6.0:amd64                1:6.0-1ubuntu2~16.04.1                     amd64        Modular compiler and toolchain technologies, runtime library
	ii  libxkbcommon0:amd64             0.5.0-1ubuntu2.1                           amd64        library interface to the XKB compiler - shared library

	
	
- I am getting torch import error
	
	- You've installed the wrong version of torch
	  
	 - Make sure that you dont use conda install to install torch
	
	 - Make sure that you installed torch=1.0.1 using pip3 or pip
		
	 - Make sure that you are running on CPU or CUDA accelarted GPU(nvidia) and check your torch version hardware choice(CPU or CUDA)

	 - Make sure that you are compiling with the right version of python==3.5.4
	   
	   Try python -V or python3 -V depending on your python interpreter

- I am getting divison by 0 error

	It's because program cant read files.
	
	- Make sure that input folder is populated with data
	
	- Makse sure that program is looking at the right path

- Loss function gives NaN

	It's called exploding gradients/vanishing gradients

	- Try to decrease the learning rate (If learning rate is too big optimzer can miss the global minimun so it diverges)

	- Try to increase weight decay and/or lambda (Could help to find the global minimum)

	- Try 1-cylce policy for learning rate : Needs implementation

	- Fit on shallow data or incerase the number of capsules and filters (If your graphs are too deep, it might be impossible to optimize since you are lossing singifanct amount of graph feature due to the misrepresentation within the capsules)

	- Try gradient clipping

	- Build a shallower network (Your network could be too deep to optimize)


- Optimizer gives an error

	It's becasue one of the training runs had a exploiding gradients problem(Loss = NaN)
	 
	 - See the above solutions






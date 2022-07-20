### Training End-to-end Steering Models with Unfiltered Driving Data

This repository contains the necessary code for training an end-to-end steering model with unfiltered data from the A2D2 dataset.

### Usage

Create a folder for the data and specify the path in the code, then run:

```python3 train.py [model] --no_itlm --no_augs --alpha=0.05```

The outputs of the training script are generated to the *outputs* directory and weights are placed in the main directory. The format for the outputted files is: ```[modelname]_[itlm/no_itlm]_[augs/no_augs]_[alpha*100].[file_format]```

To evaluate the model on the validation set and to generate samples from the validation set run the scripts:

```python3 validate.py [model]```

and

```python3 generate_samples.py [model]```


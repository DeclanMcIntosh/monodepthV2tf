# monodepthV2tf

Here is Declan and Robert's TensorFlow implementation of monodepthV2.

## Dependencies

### Models
All models can be downloaded from:
https://drive.google.com/drive/folders/14tYG4Q0djheP52v8lcKPLCZL7NT-gjjz?usp=sharing

These models should be placed directly in the source directory under the subfolder `models`.

The two models are placed as follows:
- `models/Full_data_no_mu_with_SSIM_on_left_right_only_full_loss_smoothness_0_3_disparity_scalling_res_18_bugfix_2__2020_4_19_batchsize_12/_weights_epoch20_val_loss_1.7095_train_loss_1.7080.hdf5`
- `models/Full_data_no_mu_with_SSIM_on_left_right_only_full_loss_smoothness_0_3_disparity_scalling_res_18_no_SSI__2020_4_19_batchsize_12/_weights_epoch20_val_loss_0.0662_train_loss_0.0556.hdf5`

### Python

All development was performed in Python 3.6.6 and 3.7.7 64-bit and was not tested for different versions.

### Packages
All required packages can be downloaded using PIP and the dependencies.txt file provided with this code.

## Training

### Setting Up data

Data generators have been provided but assume data is stored in the style of the Driving Stereo dataset which can be downloaded from:
https://drivingstereo-dataset.github.io/

Data was split by the corresponding zipped files, loosely corresponding to continuous runs, to ensure largest utilization of data and independence of train, validate, and test sets.

The splits were as follows by file name:

Train
- 2018-07-18-10-16-21
- 2018-07-18-11-25-02
- 2018-07-24-14-31-18
- 2018-07-27-11-39-31
- 2018-07-31-11-07-48
- 2018-07-31-11-22-31
- 2018-08-13-15-32-19
- 2018-08-13-17-45-03
- 2018-08-17-09-45-58
- 2018-10-10-07-51-49
- 2018-10-11-17-08-31
- 2018-10-12-07-57-23
- 2018-10-15-11-43-36
- 2018-10-16-07-40-57
- 2018-10-16-11-13-47
- 2018-10-16-11-43-02
- 2018-10-17-14-35-33
- 2018-10-17-15-38-01
- 2018-10-18-10-39-04
- 2018-10-18-15-04-21
- 2018-10-19-09-30-39
- 2018-10-19-10-33-08
- 2018-10-22-10-44-02
- 2018-10-23-08-34-04
- 2018-10-23-13-59-11
- 2018-10-23-15-06-54
- 2018-10-24-11-01-00
- 2018-10-24-14-13-21
- 2018-10-25-07-37-26
- 2018-10-26-15-24-18

Validate
- 2018-07-09-16-11-56
- 2018-07-10-09-54-03
- 2018-07-16-15-18-53
- 2018-07-16-15-37-46

Test 
- 2018-10-27-08-54-23
- 2018-10-27-10-02-04
- 2018-10-30-13-45-14
- 2018-10-31-06-55-01

The values for all these should be downloaded into folders corresponding to left, right, disp (disp for disparity)

The folder structure should follow

- //
    - monodepthV2tf
        - logs
        - models
    - test
        - disp
        - right
        - left
    - train
        - disp
        - right
        - left
    - val
        - disp
        - right
        - left

Corresponding folders for disp, right, and left should all have the same number of folders with corresponding identical file names as the standard set in Driving Stereo.

### Running trainer

To train the model run training.py, ensuring existence of pre-trained weights in monodepthV2tf folder, and ensuring accurate data exists.

## Evaluating 

A single script solution has been provided to evaluate the models,
The correct placement of folders with data must be provided to evaluate the model.

test.py can be run with both the provided trained models present in the monodepthV2tf directory, and existing test data. This script will print out all results described in the report. Additionally it can be given a flag to generate a random output of the network displayed with the corresponding input value for all the scales. 

This can be achieved by setting the "visualize" flag to True in this file, pressing any button on these popup windows will continue evaluating that model. Models are evaluated one after the other. 
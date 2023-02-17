# Timelog

* Song Genre Identification Tool
* Thomas McCausland
* 2472525m
* McCaig, C

## Guidance

* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.

## Week 1

### 29 Sep 2022

* *1.5 hours* Surface level research of CNNs and brainstorming how to do the project
* *0.5 hours* First supervisor meeting

### 30 Sep 2022

* *2 hours* Read part of the project guidance notes
* *0.5 hours* Created Github repository and set up local environment
* *0.5 hours* Wrote README and organised project folders

### 7 Oct 2022

* *0.5 hours* Second supervisor meeting (meeting notes in README)

### 9 Oct 2022

* *2 hours* Researched and installed Pytorch and dependencies, wrote requirements.txt for project
* *9 hours* Reasearched Pytorch tutorials and implemented a basic CIFAR10 image classification model (57% accuracy)

### 10 Oct 2022

* *6 hours* Researched more Pytorch tutorials and improved the very basic neural network  (72% accuracy)
* *2 hours* Researched nn layers and how to improve nn accuracy through other means (avoiding overfitting, training hyperparameters, etc)

### 12 Oct 2022

* *0.5 hours* Finished reading the project guidance notes
* *1 hour* Retroactively wrote timelog for the past couple weeks
* *0.5 hours* Organised project files and made temporary commit to demonstrate progress

### 13 Oct 2022

* *1 hour* Gathered minutes of last meeting to send to supervisor

### 14 Oct 2022

* *1 hour* Researched cross validation
* *0.5 hours* Third supervisor meeting
* *1 hour* Researched python spectrogram generation libraries

### 17 Oct 2022

* *5 hours* Implemented hyperparameter training
* *1 hour* Bug fixing and reading ray library docs
* *1 hour* Add option to train on preset config without hyperparameter tuning

### 18 Oct 2022

* *0.5 hours* Updated requirements.txt and set up local environment again
* *1 hour* Researched CNN architectures

### 22 Oct 2022

* *1 hour* Researched addapting CNN to have higher resolution input
* *1 hour* Various project maintenance and new script/files created for start of custom dataset generation
* *1 hour* Research into premade tagged music databases for machine learning
* *3 hours* Begin coding of dataset generation script

### 24 Oct 2022

* *1 hour* Researched existing tagged music databases (Probably going to use FMA https://github.com/mdeff/fma)

### 25 Oct 2022

* *1 hour* Fixed dependencies

### 26 Oct 2022

* *3 hours* Debugging dependencies and compatibility issues
* *2 hours* Understanding fma dataset and trying to manipulate it
* *3 hours* Writing data generation script (little progress)

### 28 Oct 2022

* *5 hours* Wrote script to generate spectrograms and selecte train, val and test sets of data

### 9 Nov 2022

* *2 hours* Wrote more script to generate spectrograms and selecte train, val and test sets of data

### 12 Nov 2022

* *5 hours* More of the same

### 13 Nov 2022

* *2 hours* Finished data_utils script 
* *3.5 hours* Converted audio data to spectrograms
* *2 hours* Researched and wrote code to load data into the CNN model (current issue with CUDA memory allocation)

### 14 Nov 2022

* *1 hour* Added custom DPI function for data loading
* *1 hour* Modified dataset_utils to load only 2 genres for proof of concept model
* *3 hour* Researched and rebuilt CNN and data transforms so that it would work (trains)

### 21 Nov 2022

* *2 hours* Re working code to be more organised and linted, adding comments for demonstration and easily accessible global variables for running tests

### 27 Nov 2022
* *1 hour* Trained hyperparameters and recorded results on binary choice case
* *1 hour* Generated full dataset for further training and plan to implement multiple binary training cases

### 4 Dec 2022
* *3 hours* Ran tests on basic CNN for different binary training cases

### 5-11 Dec 2022
* *~5 hours* Was sick and overworked during this period, forgot to keep logs. Made small bits of progress in various areas

### 12 Dec 2022
* *1 hour* Started writing report

### 15 Dec 2022
* *2 hours* Finished writing status report
* *1 hour* Cleaned up repo and made commits for long distance work over the holidays

### 9 Jan 2023
* *2 hours* Wrote script to generate genre pair directories for experiment
* *1 hour* Organised github issues to keep track of progress and organise commits
* *2 hours* Researched input transformations

### 11 Jan 2023
* *4 hours* Finished script to generate binary choice experiment data
* *3.3 hours* Ran data generation script

### 12 Jan 2023
* *1 hour* Wrote commit messages and organised github repo
* *1 hour* Various tweaks to main.py and research into ray library
* *4 hours* Various fixes to dataset_utils.py and data generation

### 16 Jan 2023
* *2 hours* Research into rayTune CLIReporter class
* *7 hours* Wrote code for running binary choice experiment training and refactored a lot of code

### 18 Jan 2023
* *4 hours* Wrote code to finish up experiment_run() method and output data to csv file
* *2 hours* Fixed various bugs
* *1 hour* Trained first experiment on limited set of 3 genres

### 21 Jan 2023
* *3 hours* Wrote more code to main.py

### 23 Jan 2023
* *6 hours* Wrote code to validate dataset
* *4 hours* Ran experiment on 5 genre set

### 2 Feb 2023
* *7 hours* Wrote python script to load trained models and evaluate them

### 4 Feb 2023
* *2 hours* Fixed data generation script for full 8 genres
* *3 hours* Modified model evaluation script for full 8 genres and refactored a few methods
* *2 hours* Fixed and ran model training script

### 5 Feb 2023
* *2 hours* Wrote script for generating multiclass confusion matrix
* *2 hours* Figured out how to get true labels of data

### 6 Feb 2023
* *28 hours* Hyperparameter training, found best config

### 10 Feb 2023
* *4 hours* Finished evaluation script

## Year 4 Individual Project

# The Spectrogram One

### Project:
    Song Genre Identification Tool
### Supervisor:
    McCaig, C
### Description:
    Song genre is an important piece of metadata within MP3s. Sometimes tracks can be produced in a way that leads to the genre being absent from the MP3s metadata. This project will involve developing a machine learning model that can “listen” to an excerpt of a song and identify the genre(s) it belongs to. The model can be trained using a labelled dataset of song samples (e.g. GTZAN). With a suitably trained model  develop a tool to identify the genre of a song and add the appropriate information to the metadata of the track.

***



***

### Initial Project Checklist (as of 30/09/2022)
- Work on pytorch tutorials for machine learning models (identifying cats and dogs)
- Build simple preliminary database of song spectrograms with genre tags
- Write simple script to turn song snippets into spectrograms
- Research convolutional neural networks for image recognition
- Consider how metadata can be used to refine the degree of certainty
- Investigate Spotify API


***

### Project Diary:
- 29/09/2022 First Supervisor Meeting
- 30/09/2022 Created project repo
- 07/10/2022 Second Supervisor Meeting
    - Notes:
        - Discussed pytorch tutorials
        - Start model with binary genre identification
        - relatively small starting dataset, couple hundred samples
        - No loss of data from conversion to spectrogram -> to investigate
        - Investigate spotify api when the time comes, gather initial database by hand if necessary
        - Asked for the project proposal description
- 14/10/2022 Third Supervisor Meeting
    - Notes:
        - Weekly status reports: quick summary, this is what I did and what I plan on doing (headlines), any pressing questions 
        - Look into a downloadable dataset of audio data
        - Look into other similar projects
        - Consider simpler, uniform songs first
        - Consider generating visual patterns based on trained model to see the emerging patterns
        - Research standards for spectrograms and spectrogram generation (logarithmic scale, miel scale)
        - Test out different resolutions of spectrogram (trade off between training time and data loss)
        - 100 datapoints is good to start with, check against random chance and if it's promissing, improve the dataset
28/10/2022 Fourth Supervisor Meeting
    - Notes:
        - How to save spectrograms: png but compression, try binary if possible
        - Spectrogram resolution is fine
        - Chris will send help with pandas
07/11/2022 Fifth Supervisor Meeting
    - Notes:
        - Use ImageNet dataset example/tutorials for loading in data
        - file tree structure to sort data into subsets (train/rock/ etc)
        - research format of input into model
        - could as an extension compare performance with a resnet model (for diss)
        - compare other research done on this topic (for diss)
14/11/2022 Sixth Supervisor Meeting
    - Notes:
        - Servers available at Uni (stlinux12 and stlinux13 have GPUs)
        - Start with fewer genres then expand (binary choice)
        - Lower resolution of input data, perhaps 256x256
        - both should help fix memory issues
21/11/2022 Seventh Supervisor Meeting
    - Notes:
        - Presented results, discussed next steps
        - Binary choice good so far, to add more data classes and see how it performs
        - Tune the model better
        - Not worry about the resolution of the input
        - Consider running tests on other binary class sets
        - Consider examining auxiliary genres and running binary test on most closely related top-genres
            - expected to see in diss, confusion matrix
28/11/2022 Eighth Supervisor Meeting
    - Notes:
        - Presented results, discussed next steps
        - Next, look at improving the model and dataset
        - Look at output and set custom cutoff
        - Research origin of database, how are top_genres decided (for diss discussion)
        - Encouraged to explore using ResNet
        - Definitely implement dropout layers and other techniques
        - Look at ResNet and use as inspiration for current custom model (gives things to talk about in diss)
        - Don't forget status report at end of semester
12/12/2022 Ninth Supervisor Meeting
    - Notes:
        - Discussed end of semester report
        - Outline Dissertation
        - This is a novel piece of work -> good to highlight in diss
        - Chris offered to help publish this work, very exciting offer
        - Look at other literature, see what others have done, put my work into context
        - Document process, choices, dissertation can balance out grade if project falls short
        - Think of shape of dissertation, get stuff written down

***

#### Sources & Materials

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
https://www.youtube.com/watch?v=d9QHNkD_Pos&ab_channel=freeCodeCamp.org
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
https://towardsdatascience.com/choosing-a-hyperparameter-tuning-library-ray-tune-or-aisaratuners-b707b175c1d7
https://docs.ray.io/en/latest/ray-air/package-ref.html
https://www.jeremyjordan.me/convnet-architectures
https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
https://arxiv.org/abs/1207.0580
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
https://www.youtube.com/watch?v=3gzI4Z2OFgY&ab_channel=ValerioVelardo-TheSoundofAI
https://github.com/musikalkemist/AudioSignalProcessingForML/blob/master/16%20-%20Extracting%20Spectrograms%20from%20Audio%20with%20Python/Extracting%20Spectrograms%20from%20Audio%20with%20Python.ipynb
https://music-classification.github.io/tutorial/part2_basics/input-representations.html
https://arxiv.org/abs/1612.01840
https://librosa.org/doc/latest/install.html
https://paperswithcode.com/dataset/fma
https://github.com/mdeff/fma
https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
https://pytorch.org/docs/stable/notes/faq.html


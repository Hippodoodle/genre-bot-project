
## *Song Genre Identification Tool* 
#### *Thomas McCausland* 
#### *2472525* 

## Proposal
### Motivation
*[Clearly motivate the purpose of your project; why someone would care about what you are doing]*
Song genre is an important piece of metadata within MP3s. Sometimes tracks can be produced in a way that leads to the genre being absent from the MP3s metadata. This project will involve developing a machine learning model that can “listen” to an excerpt of a song and identify the genre(s) it belongs to. The model can be trained using a labelled dataset of song samples (e.g. GTZAN). With a suitably trained model  develop a tool to identify the genre of a song and add the appropriate information to the metadata of the track.


### Aims
*[Clearly state what the project is intended to do. This should be something which is measurable; it should be possible to tell if you succeeded]*
This project aims to research and develop a machine learning model capable of identifying and classifying song genre, using image recognition. The minimum viable product is a model that has "better than random chance" testing accuracy.


## Progress
*[Briefly state your progress so far, as a bulleted list]*
- Developed a simple convolutional neural network model using pytorch for image recognition on the CIFAR-10 dataset
- Researched convolutional neural networks and typical machine learning model layers
- Improved the simple CNN model
- Researched and developed script to convert song files into spectrograms
- Researched and chose an appropriate existing song dataset labelled by genre: FMA dataset
- Wrote script to generate custom image dataset of song spectrograms
- Researched how to adapt the developed CNN model for use on custom image dataset
- Adapted the CNN model for use on custom spectrogram dataset and ran tests to test its initial accuracy. Better than random chance testing accuracy has been achieved.


## Problems and risks
### Problems
*[What problems have you had so far, that have held up the project?]*
- 


### Risks
*[What problems do you foresee in the future and how will you mitigate them?]*


## Plan
*[Time plan, in roughly weekly to monthly blocks, up until submission week]*



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

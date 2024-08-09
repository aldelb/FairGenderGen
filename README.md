

**To see generation examples with natural voices, go [here](https://www.youtube.com/playlist?list=PLRyxHB7gYN-D8v_mMn4RZvJfH00h7bRT1).**

# Mitigation of gender bias in automatic facial non-verbal behaviours generation
The code contains two models 
* the first one to jointly and automatically generate the rhythmic head, facial, and gaze movements (non-verbal behaviors) of a virtual agent from acoustic speech features. The architecture is an Adversarial Encoder-Decoder. Head movements and gaze orientation are generated as 3D coordinates, while facial expressions are generated using action units based on the facial action coding system.
* The second, building upon the first, integrates a gender discriminator and a gradient reversal layer. It aims to mitigate these biases and create non-verbal behaviours independent of the speaker’s gender


The github repository of the gender classifier is available [here](https://github.com/behavioursGeneration/gender-classifier?tab=readme-ov-file).
  
## The architecture 
![Capture](https://github.com/behavioursGeneration/FairGenderGen/assets/110098017/da6d223d-f38e-44b9-9529-32eaacfdfbcf)

### Features recovery 
With this section, we directly recover the extracted and align features. We extract the speech and visual features automatically from videos using existing tools, namely OpenFace and OpenSmile. 

You can use the code in the [pre_processing](https://github.com/behavioursGeneration/FairGenderGen/tree/main/pre_processing) folder to extract your own features from chosen videos.
1. Extract both behavioural and speech features
```
python pre_processing/extract_openface.py trueness_1 True
python pre_processing/extract_opensmile.py trueness True
```
OR 

Contact the authors to obtain the features extracted et processed from the dataset Trueness. 

### Models training
1. "params.cfg" is the configuration file to customize the model before training [the configuration file](docs/params_base.cfg).
2. You can conserve the existing file or create a new one. 
3. In the conda console, train the model by executing:
```
python PATH/TO/PROJECT/generation/main.py -params PATH/TO/CONFIG/FILE.cfg [-id NAME_OF_MODEL] -task train
```
You can visualize the created graphics during training in the repository [saved_path] of your config file. 

### Behaviours generation
In the conda console, generate behaviors by executing:
```
python PATH/TO/PROJECT/generation/main.py -task generate -params PATH/TO/CONFIG/FILE.cfg -epoch [integer] -dataset [dataset]
```
The behaviors are generated in the form of 3D coordinates and intensity of facial action units. These are .csv files stored in the repository [output_path] of your config file. 

- -epoch: during training, if you trained in 1000 epochs, recording every 100 epochs, you must enter a number within [100;200;300;400;500;600;700;800;900;1000].
- -params: path to the config file. 
- -dataset: name of the considered dataset. 


### Animate the generated behaviors
To animate a virtual agent with the generated behaviors, we use the GRETA platform. 

1. Download and install GRETA with "gpl-grimaldi-release.7z" at https://github.com/isir/greta/releases/tag/v1.0.1.
2. Open GRETA. Open the configuration "Greta - Record AU.xml" which is already present in GRETA. 
3. Use the block "AU Parser File Reader" and "Parser Capture Controller AU" to create the video from the .csv file generated.

### Add voice 
You can directly concatenate the voices from the original videos to the Greta generated .avi videos.

```
input_video = "video_path.avi"
input_audio = "audio_path.wav"

output = "video_path_with_sound.mp4"

if(os.path.isfile(input_video) and os.path.isfile(input_audio)):
     audio = mp.AudioFileClip(input_audio)
     video = mp.VideoFileClip(input_video)
     final = video.set_audio(audio)

     final.write_videofile(output)
```




# SVA
## Getting started

### Installation

**1. AudioCraft**

This project heavily rely on the AudioCraft by Meta. The environment requirements, model checkpoints and all  necessary information can be found at https://github.com/facebookresearch/audiocraft. Just follow it and make sure you can run the demos of AudioGen and MusicGen successfully. 

**2. FFmpeg**

FFmpeg is neccesary for video & audio editing. Go to https://ffmpeg.org/ for the latest version. We recommend using static builds which can be downloaded at https://johnvansickle.com/ffmpeg/. In our implementation, the verision is **"N-68911-ge70e9b6554-static"**.

**3. Our codes**

Git clone our repository, using the python environment you have created during the AudioCraft installation.

```
git clone https://github.com/HUIZ-A/SVA.git
```

### Launching

You can run a demo command for a quick start in a default setting:

```
cd SVA
python run.py --video_path "./RawVideo/man-on-the-cloud.mp4" 
```

Your can also specify your requirements and device by adding arguments `--user_input` and `--device`:

```
python run.py --video_path "./RawVideo/man-on-the-cloud.mp4" --user_input "i wanna a cyberpunk and future tech style" --device "cuda:0"
```

Check the final results and intermediate files by running:

```
cd results/<Your Video Name>
```

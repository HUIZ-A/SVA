# SVA: Semantically consistent Video-to-Audio Generation
paper: [[2404.16305\] Semantically consistent Video-to-Audio Generation using Multimodal Language Large Model (arxiv.org)](https://arxiv.org/abs/2404.16305)

project: [Semantically consistent Video-to-Audio Generation using Multimodal Language Large Model (huiz-a.github.io)](https://huiz-a.github.io/audio4video.github.io/)



SVA is a light and efficient framework for video-to-audio generation, comprising the following key steps:.

- Utilizing MLLM for comprehending video content and generating audio and music schemes.
- Employing generation models to produce audio or music in line with the given schemes.
- Incorporating fast noise detection, noise reduction, and mixing methods to generate high-quality videos with audio.



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

**4. API Keys**

The MLLM **Gemini-Pro-1.0** by Google is used for video understanding and scheme generation. Before running the codes, make sure you own the API Key and input it in the `pipeline.py` line 18.

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

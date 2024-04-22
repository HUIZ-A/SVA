import os
import subprocess
import json
from gradio_client import Client
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import requests
import PIL.Image
import base64
import torchaudio
from audiocraft.models import MusicGen
import librosa
import numpy as np
import glob
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import argparse

API_KEY = 'AIzaSyAx8tnQZ-PzfSxCQdQkbmfMXVr-JbF2PCM'

def get_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--video_path", type=str,help="path to target video file.")
    parser.add_argument("--style", type=str, default='creative', help="specify the idea style.")
    args = parser.parse_args()
    return args

# 向gemini post query，接收返回文本。可切换多模态
def get_gemini_txt(query, version):
    body = query
    # 定义请求头
    headers = {
        'Content-Type': 'application/json',
    }
    if version == 'vision':
        model = "gemini-pro-vision"
    elif version == 'txt':
        model = "gemini-pro"
    else:
        raise Exception('version is either txt or vision')

    response = requests.post(
        'https://gemini.cghdfyyyw.work/v1beta/models/{}:generateContent?key={}'.format(model, API_KEY),
        headers=headers,
        json=body,
    )

    # 检查响应状态码
    if response.status_code == 200:
        # 解析响应数据
        data = response.json()
        # print("@",data)

        result_list = []

        # 打印响应内容
        for candidate in data['candidates']:
            result_list.append(candidate['content']['parts'][0]['text'])
    else:
    # 打印错误信息
        print('Error:', response.status_code, response.text)
    return result_list[0]


# 执行ffmpeg命令，返回p,err
def subprocess_cmd(cmd:str):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    out, err = p.communicate()
    return out, err


# 对视频进行关键帧切片，存储在中间目录
def clip(video_path):
    ext = os.path.splitext(video_path)[-1]
    if ext != '.mp4' and ext != '.avi' and ext != '.flv':
        raise Exception('format not support')
    
    full_file_name = os.path.split(video_path)[-1]
    file_name = os.path.splitext(full_file_name)[0]
    pic_folder = f'/new_data/gehui/projects/audio4video/{file_name}/pic'
    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)
    elif len(glob.glob(os.path.join(pic_folder, '*.jpg')))>0:
        print("clips are already stored in:", pic_folder)
        return pic_folder
        print('????')
    else:
        raise Exception('empty pic_folder or wrong folder:',pic_folder)

    cmd = f'ffmpeg -i {video_path} -vf "select=eq(pict_type\,I)"  -vsync vfr -qscale:v 2 -f image2 {pic_folder}/%08d.jpg'
    
    p = subprocess.Popen(
        cmd.format(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)

    out, err = p.communicate()
    # print("subprocess 执行结果：out:{} err:{}".format(out, err))
    return pic_folder


def caption(video_path):
    full_video_name = os.path.split(video_path)[-1]
    video_name = os.path.splitext(full_video_name)[0]    
    pic_folder = f'/new_data/gehui/projects/audio4video/{video_name}/pic'
    
    if os.path.exists(f'./{video_name}/caption.txt'):
        with open(f'./{video_name}/caption.txt', 'rb') as f:
            caption = f.read()
        print("caption is already here:", caption)
        return caption

    # 将图片转换为 base64 编码
    with open(f"{pic_folder}/00000001.jpg", "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # 将 base64 编码转换为字符串，不包含换行符
    image_base64_str = image_base64.replace("\n", "")

    query = {
        "contents":[
            {
            "parts":[
                {"text": "describe the video based on the picture captured from it"},
                {
                "inline_data": {
                    "mime_type":"image/jpeg",
                    "data": f'{image_base64_str}'
                }
                }
            ]
            }
        ]
    }
    
    caption = get_gemini_txt(query, "vision")
    
    
    with open(f'./{video_name}/caption.txt', 'w') as f:
        f.write(caption)
    return caption


def plan(video_path, caption, style)->dict:
    template_blue = f"""
        BGM stands for Background Music. It refers to the musical soundtrack or audio accompaniment that plays in the background of various media, such as films, TV shows, video games, presentations, and other content. The purpose of background music is to enhance the mood, atmosphere, or emotional impact of a scene without overshadowing the primary audio elements like dialogue and sound effects. BGM can range from subtle and ambient to more prominent and thematic, depending on the context and desired effect in the media it accompanies.

SFX stands for Sound Effects. It refers to all the sounds in a film, video game, or other media besides dialogue and music. These sounds can be:
​  Natural sounds: Things that exist in the real world, like wind, rain, thunder, or animal noises.
​  Artificial sounds: Sounds created by humans, like explosions, gunshots, car engines, or footsteps.
​  Foley sounds: Sounds that are specifically created to match the visuals on screen, often done live during filming. For example, someone might crumple a bag to create the sound of footsteps on snow.

You are good at coming up BGM and SFX ideas for a video based on its description. Here is the description of a video: {caption}

Now plan blue and sad SFXs and BGM, in order to express blue and sorrow for the audiences, even though the video description is not sad.

The SFX output is one short sentence in 12 words, which is not concerning music and instruments, and must be common in real world. If the sound is unusal and novel, find some similar and normal sound to replace it,like:
​  'Sirens and a humming engine approach and pass';
​  'A duck quacking as birds chirp and a pigeon cooing';
​  'Railroad crossing signal followed by a train passing and blowing horn';
​  'Typing on a typewriter'

The BGM output is a brief description, involving instruments, beats, melody, mood, style and so on,like:

​  'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle' ,;
​  'A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city',;
​  'Smooth jazz, with a saxophone solo, piano chords, and snare full drums',;
​  '90s rock song with electric guitar and heavy drums'. 

Now output one SFX and BGM idea that blue, sorrowful and sad, comprising one BGM and 2 SFXs. Output the idea following the examples below in json, do not use any brackets and single quotes '' in a string:       
    """
    template_creative = f"""
        BGM stands for Background Music. It refers to the musical soundtrack or audio accompaniment that plays in the background of various media, such as films, TV shows, video games, presentations, and other content. The purpose of background music is to enhance the mood, atmosphere, or emotional impact of a scene without overshadowing the primary audio elements like dialogue and sound effects. BGM can range from subtle and ambient to more prominent and thematic, depending on the context and desired effect in the media it accompanies.
        SFX stands for Sound Effects.  It refers to all the sounds in a film, video game, or other media besides dialogue and music.  These sounds can be:
        Natural sounds: Things that exist in the real world, like wind, rain, thunder, or animal noises.
        Artificial sounds: Sounds created by humans, like explosions, gunshots, car engines, or footsteps.
        Foley sounds: Sounds that are specifically created to match the visuals on screen, often done live during filming. For example, someone might crumple a bag to create the sound of footsteps on snow.
        
        You are good at coming up BGM and SFX ideas for a video based on its description. Here is the description of a video: {caption}
        Now Plan creative, Wild and imaginative, Blue-sky thinking and Out of this world SFXs and BGM, in order to express innovation and amusement for the audiences.
        The SFX output is one short sentence in 12 words, which is not concerning music and instruments, and must be common in real world. If the sound is unusal and novel, find some similar and normal sound to replace it,like:


        'Sirens and a humming engine approach and pass';
        'A duck quacking as birds chirp and a pigeon cooing';
        'Railroad crossing signal followed by a train passing and blowing horn';
        'Typing on a typewriter'

        The BGM output is a brief description, involving instruments, beats, melody, mood, style and so on,like:
        'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle' ,;
        'A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city',;
        'Smooth jazz, with a saxophone solo, piano chords, and snare full drums',;
        '90s rock song with electric guitar and heavy drums'. 
 
        Now output one SFX and BGM idea that creative, Wild and imaginative, Blue-sky thinking and Out of this world, comprising one BGM and 2 SFXs. Output the idea following the examples below in json, do not use any brackets and single quotes '' in a string:
    """
    template_normal = f"""
        BGM stands for Background Music. It refers to the musical soundtrack or audio accompaniment that plays in the background of various media, such as films, TV shows, video games, presentations, and other content. The purpose of background music is to enhance the mood, atmosphere, or emotional impact of a scene without overshadowing the primary audio elements like dialogue and sound effects. BGM can range from subtle and ambient to more prominent and thematic, depending on the context and desired effect in the media it accompanies.
        SFX stands for Sound Effects.  It refers to all the sounds in a film, video game, or other media besides dialogue and music.  These sounds can be:
        Natural sounds: Things that exist in the real world, like wind, rain, thunder, or animal noises.
        Artificial sounds: Sounds created by humans, like explosions, gunshots, car engines, or footsteps.
        Foley sounds: Sounds that are specifically created to match the visuals on screen, often done live during filming. For example, someone might crumple a bag to create the sound of footsteps on snow.
        You are good at coming up BGM and SFX ideas for a video based on its description. Here is the description of a video: {caption}
        Now Plan proper SFXs and BGM which suites for the given video's style, in order to make the video attractive and impressive for the audiences.
        The SFX output is one short sentence in 12 words, which is not concerning music and instruments, and must be common in real world. If the sound is unusal and novel, find some similar and normal sound to replace it,like:
        'Whistling with wind blowing';
        'Sirens and a humming engine approach and pass';
        'A duck quacking as birds chirp and a pigeon cooing';
        'Railroad crossing signal followed by a train passing and blowing horn';
        'Typing on a typewriter'

        The BGM output is a brief description, involving instruments, beats, melody, mood, style and so on,like:
        'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle' ,;
        'A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city',;
        'Smooth jazz, with a saxophone solo, piano chords, and snare full drums',;
        '90s rock song with electric guitar and heavy drums'. 
 
        Now output one SFX and BGM idea the most appropriate and sensible, comprising one BGM and 2 SFXs. Output the idea following the examples below in json, do not use any brackets and single quotes '' in a string:              
    """

    template_warm = f"""
        BGM stands for Background Music. It refers to the musical soundtrack or audio accompaniment that plays in the background of various media, such as films, TV shows, video games, presentations, and other content. The purpose of background music is to enhance the mood, atmosphere, or emotional impact of a scene without overshadowing the primary audio elements like dialogue and sound effects. BGM can range from subtle and ambient to more prominent and thematic, depending on the context and desired effect in the media it accompanies.

SFX stands for Sound Effects. It refers to all the sounds in a film, video game, or other media besides dialogue and music. These sounds can be:
​  Natural sounds: Things that exist in the real world, like wind, rain, thunder, or animal noises.
​  Artificial sounds: Sounds created by humans, like explosions, gunshots, car engines, or footsteps.
​  Foley sounds: Sounds that are specifically created to match the visuals on screen, often done live during filming. For example, someone might crumple a bag to create the sound of footsteps on snow.

You are good at coming up BGM and SFX ideas for a video based on its description. Here is the description of a video: {caption}

Now plan positive, warm and sunny style SFXs and BGM, in order to express happiness and warmness for the audiences, even though the video description is sad.

The SFX output is one short sentence in 12 words, which is not concerning music and instruments, and must be common in real world. If the sound is unusal and novel, find some similar and normal sound to replace it,like:
​  'Sirens and a humming engine approach and pass';
​  'A duck quacking as birds chirp and a pigeon cooing';
​  'Railroad crossing signal followed by a train passing and blowing horn';
​  'Typing on a typewriter'

The BGM output is a brief description, involving instruments, beats, melody, mood, style and so on,like:

​  'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle' ,;
​  'A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city',;
​  'Smooth jazz, with a saxophone solo, piano chords, and snare full drums',;
​  '90s rock song with electric guitar and heavy drums'. 

Now output one SFX and BGM idea that happy, warm and sunny, comprising one BGM and 2 SFXs. Output the idea following the examples below in json, do not use any brackets and single quotes '' in a string:       
    """

    examples_normal = """{"idea":"Mystical Curiosity","SFX":["High-pitched wind chime tinkling softly","Distant owl hooting softly"],"BGM":"A whimsical and playful piece with a glockenspiel melody, light percussion using woodblocks and triangles, and a backdrop of ethereal chimes"}
    {"idea":"Prehistoric Dance Party", "SFX":["Stomping mammoth feet shaking the ground", "High-pitched trumpet calls from the mammoths"],"BGM":"Upbeat electronic dance music with a strong bassline and prehistoric-inspired synth sounds"}
    {"idea":"City Symphony","SFX":["High heels clicking on pavement","Distant laughter and chatter"],"BGM":"A dynamic blend of hip-hop and orchestral elements, with a driving beat, pulsing bassline, and sweeping strings and brass, capturing the vibrant energy of the city"}
    
    """
    
    examples_blue = """{"idea":"Haunting Loneliness","SFX":["Wind whistling mournfully through empty streets","Distant, lonely howl of a stray dog"],"BGM":"A slow and melancholic piano piece in a minor key, punctuated by occasional, dissonant chords"}
    {"idea":"Echoes of Lost Dreams", "SFX":["Faint, distorted murmurs of a forgotten conversation", "Raindrops gently tapping a melancholic rhythm on a window pane"],"BGM":"A haunting orchestral piece with muted strings and woodwinds, featuring dissonant melodies that fade in and out"}
    {"idea":"Neon Dreams Fading","SFX":["Flickering neon sign buzzing erratically, casting a dying glow","Footsteps echoing hollowly in a deserted alleyway"],"BGM":"A downtempo electronic track with a melancholic melody, distorted synths, and a slow, fading beat"}
    """

    examples_warm = """{"idea":"Sparkling Discovery","SFX":["High-pitched wind chime tinkling playfully in a gentle breeze","chirping of crickets on a warm summer night"],"BGM":"A whimsical and upbeat piece with a glockenspiel melody"}
    {"idea":"Prehistoric Frolicking", "SFX":["Playful trumpeting sounds from young mammoths frolicking in a field", "the gentle thumps of their footsteps as they chase butterflies"],"BGM":"Upbeat and cheerful electronic dance music with a moderate bassline and prehistoric-inspired synth melodies that evoke a sense of wonder and joy"}
    {"idea":"Urban Symphony of Joy","SFX":[" Upbeat and lively street music spilling out from a nearby cafe","happy chatter and laughter from people enjoying a sunny day"],"BGM":"A bright and energetic blend of pop and orchestral elements, with a driving beat"}
    """

    if style == 'normal':
        template = template_normal
    elif style == 'creative':
        template = template_creative
    elif style == 'blue':
        template = template_blue
        examples = examples_blue
    elif style == 'warm':
        template = template_warm
        examples = examples_warm
    else:
        raise Exception("style must be 'normal' or 'creative'")

    template += template + examples
    # print("!!!",template)

    query = {
        "contents":[
            {
            "parts":[
                {"text": f"{template}"},
            ]
            }
        ]
    }
    plan = get_gemini_txt(query, 'txt')
    plan = eval(plan)
    plan_json = json.dumps(plan)
    full_file_name = os.path.split(video_path)[-1]
    file_name = os.path.splitext(full_file_name)[0]
    title = plan['idea'].replace(" ","")
    plan_folder = f'./{file_name}/{title}'
    if not os.path.exists(plan_folder):
        os.makedirs(plan_folder)
    with open(f'{plan_folder}/plan.json', "w") as f:
        f.write(plan_json)
    
    return plan

def get_video_duration(video_path: str):
    ext = os.path.splitext(video_path)[-1]
    if ext != '.mp4' and ext != '.avi' and ext != '.flv':
        raise Exception('format not support')
    ffprobe_cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'
    p = subprocess.Popen(
        ffprobe_cmd.format(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    out, err = p.communicate()
    print("duration 执行结果：out:{}".format(out))
    duration_info = float(str(out, 'utf-8').strip())
    return duration_info

def get_SFX(video_path, plan_dict):
    full_file_name = os.path.split(video_path)[-1]
    file_name = os.path.splitext(full_file_name)[0]
    title = plan_dict['idea'].replace(" ","")
    plan_folder = f'./{file_name}/{title}'

    sfx_0 = f'{plan_folder}/0.wav'
    sfx_1 = f'{plan_folder}/1.wav'
    if os.path.exists(sfx_1) and os.path.exists(sfx_0):
        print('already exists 2 wavs, skip')
        return

    model = AudioGen.get_pretrained('/new_data/gehui/MyModels/audiogen-medium')
    
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=get_video_duration(video_path)
    )


    wav = model.generate(
        descriptions=plan_dict['SFX'],
        progress=True
    )

    full_file_name = os.path.split(video_path)[-1]
    file_name = os.path.splitext(full_file_name)[0]
    title = plan_dict['idea'].replace(" ","")

    for idx, one_wav in enumerate(wav):
        path = audio_write(f'/new_data/gehui/projects/audio4video/{file_name}/{title}/{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    return path


def get_BGM(video_path, plan_dict):
    device = 'cuda:2'
    processor = AutoProcessor.from_pretrained("/new_data/gehui/MyModels/MusicGen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("/new_data/gehui/MyModels/MusicGen-medium").to(device)

    inputs = processor(
        text=[plan_dict['BGM']],
        padding=True,
        return_tensors="pt",
    ).to(device)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=round(get_video_duration(video_path)*50+0.5)).cpu()
    sampling_rate = model.config.audio_encoder.sampling_rate

    full_file_name = os.path.split(video_path)[-1]
    file_name = os.path.splitext(full_file_name)[0]
    title = plan_dict['idea'].replace(" ","")

    scipy.io.wavfile.write(f"/new_data/gehui/projects/audio4video/{file_name}/{title}/BGM.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
    return f"/new_data/gehui/projects/audio4video/{file_name}/{title}/BGM.wav"
   


def is_noise(audio_path, threshold=0.3):
    # 读取音频文件
    y, sr = librosa.load(audio_path)

    # 计算音频的短时傅里叶变换
    stft = np.abs(librosa.stft(y))

    # 计算每帧的能量
    energy = librosa.feature.rms(S=stft)

    # 判断是否存在噪音
    avg_energy = np.mean(energy)
    return avg_energy > threshold


def reduce_noise(audio_path:str)->str:
    ext = os.path.splitext(audio_path)[-1]
    audio_folder = os.path.dirname(audio_path)
    full_audio_name = os.path.split(audio_path)[-1]
    audio_name = os.path.splitext(full_audio_name)[0]
    if ext != '.wav':
        raise Exception('format not support')
    cmd = f'ffmpeg -i {audio_path} -af lowpass=3000,highpass=200,afftdn=nf=-25 -channel_layout mono -y {audio_folder}/{audio_name}_1.wav'
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    out, err = p.communicate()
    # print(out, err)

    return f'{audio_folder}/{audio_name}_1.wav'

def mix(video_path:str, plan_dict:dict)->str:
    full_video_name = os.path.split(video_path)[-1]
    video_name = os.path.splitext(full_video_name)[0]
    title = plan_dict['idea'].replace(" ","")
    plan_folder = f'./{video_name}/{title}'
    # 默认2条sfx
    sfx_0 = f'./{plan_folder}/0.wav'
    sfx_1 = f'./{plan_folder}/1.wav'
    sfxs = [sfx_0, sfx_1]
    if len(sfxs) != 2:
        raise Exception('sfx files are not 2')
    # 丢弃低质量sfx
    remain_sfx = []
    for i in sfxs:
        if not is_noise(i):
            remain_sfx.append(i)
            print(f"sfx{i} passed the low quality filter")
    # BGM去噪
    bgm = f'{plan_folder}/BGM.wav'
    bgm_1 = reduce_noise(bgm)
    # 移除原视频音频流
    vonly_cmd = f'ffmpeg -i {video_path} -c:v copy -an {plan_folder}/videoWithoutAudio.mp4'
    out,err = subprocess_cmd(vonly_cmd)
    # print("vonly err",err)
    # 混音
    sfx_num = len(remain_sfx)
    bgm_num = 0
    for file in os.listdir(f'{plan_folder}'):
        if file == 'BGM_1.wav':
            bgm_num += 1
    if sfx_num > 2 and bgm_num != 1:
        raise Exception('sfx or bgm num error')
    print("sfx_num is", sfx_num)
    if sfx_num == 0:
        mix_cmd = f"""
            ffmpeg -i {plan_folder}/videoWithoutAudio.mp4 -i {plan_folder}/BGM_1.wav -map 0 -map 1:a -c:v copy -c:a aac -b:a 192k -shortest {plan_folder}/final.mp4   
        """
    elif sfx_num == 1:
        mix_cmd = f"""
            ffmpeg -i {plan_folder}/videoWithoutAudio.mp4 -i {remain_sfx[0]} -i {plan_folder}/BGM_1.wav -filter_complex "[1:a:0]volume=0.05[a1];[a1][2:a:0]amix=inputs=3.0:normalize=0:duration=shortest[aout]" -c:v copy -map 0:v:0 -map [aout] -c:a aac -b:a 192k -channel_layout mono -shortest -y {plan_folder}/final.mp4
        """
    elif sfx_num == 2:
        mix_cmd = f"""ffmpeg -i {plan_folder}/videoWithoutAudio.mp4 -i {sfx_0} -i {sfx_1} -i {plan_folder}/BGM_1.wav -filter_complex "[1:a:0]volume=0.05[a1];[2:a:0]volume=0.05[a2];[3:a:0]volume=3.0[a3];[a1][a2][a3]amix=inputs=3:normalize=0:duration=shortest[aout]" -c:v copy -map 0:v:0 -map [aout] -c:a aac -b:a 192k -channel_layout mono -shortest -y {plan_folder}/final.mp4"""
    else:
        raise Exception('sfx or bgm num error')
    print("mix_cmd:", mix_cmd)
    # 执行命令
    out, err = subprocess_cmd(mix_cmd)

    return f'{plan_folder}/final.mp4'
    
if __name__ == '__main__':
    args = get_args()
    video_path = args.video_path
    style = args.style
    # plan_dict = {"idea": "Mammoths in the Snow", "SFX": ["Light crunching of snow under heavy feet", "Mammoths' deep breathing and stomping"], "BGM": "A sweeping orchestral piece with a slow, steady tempo, featuring strings and low brass to evoke a sense of grandeur and the vastness of the snowy landscape"}
    print("1.clip the video")
    pic_folder = clip(video_path)
    print("clips are stored in: ", pic_folder)
    
    print("2.caption the video")
    video_caption = caption(video_path)
    print("the video caption is:", video_caption)

    print("3.plan SFX and BGM")
    plan_dict = plan(video_path, video_caption, style)
    print("ideas are: ", plan_dict)

    print("4.generate SFX")
    get_SFX(video_path, plan_dict)
    print("done")

    print("5.generate BGM")
    get_BGM(video_path, plan_dict)
    print("done")

    print("6.mix")
    result_path = mix(video_path, plan_dict)
    print("result", result_path)






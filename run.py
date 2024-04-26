from pipeline import *

def get_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--video_path", type=str,help="path to target video file.")
    parser.add_argument("--user_input", type=str, default=None, help="user personalization prompt.")
    parser.add_argument("--device", type=str, default='cuda:0', help="specify the gpu.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    video_path = args.video_path
    user_input = args.user_input
    device = args.device
    
    pipe = SVAPipeline(device)
    result_path = pipe(video_path, user_input)
    print(f"The result is stored at", result_path)
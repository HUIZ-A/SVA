for i in {1..10}
do
    echo "Running Python program iteration $i"
    python audio4video.py \
        --video_path "./RawVideo/monster-with-melting-candle.mp4"\
        --style "creative"  # 将 "your_program.py" 替换为你的 Python 程序文件名
done
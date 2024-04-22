# 获取文件夹路径
folder_path=$"./RawVideo"

# 遍历指定文件夹及其子目录下的所有MP4文件
find "$folder_path" -type f -name "*.mp4" | while read -r file; do
    # 打印文件路径
    echo "Processing file: $file"

    # 调用Python程序，并传递文件路径作为参数
    python audio4video.py \
        --video_path "$file"\
        --style "warm"

    echo -e "Processing done"
done
folder_path=$"./RawVideo"

# 遍历指定文件夹及其子目录下的所有MP4文件
find "$folder_path" -type f -name "*.mp4" | while read -r file; do
    # 打印文件路径
    echo "Processing file: $file"

    # 调用Python程序，并传递文件路径作为参数
    python audio4video.py \
        --video_path "$file"\
        --style "warm"

    echo -e "Processing done"
done
folder_path=$"./RawVideo"

# 遍历指定文件夹及其子目录下的所有MP4文件
find "$folder_path" -type f -name "*.mp4" | while read -r file; do
    # 打印文件路径
    echo "Processing file: $file"

    # 调用Python程序，并传递文件路径作为参数
    python audio4video.py \
        --video_path "$file"\
        --style "warm"

    echo -e "Processing done"
done
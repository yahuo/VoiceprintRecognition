#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 打印帮助信息
function show_help {
    echo "========================================================"
    echo "  🎙️  FunASR 声纹识别与会议记录工具箱"
    echo "========================================================"
    echo "用法:"
    echo "  ./run.sh register [姓名] [音频路径]    - 注册声纹"
    echo "  ./run.sh meeting [音频路径]            - 生成会议记录"
    echo "  ./run.sh live [输出文件]               - 🔴 实时会议记录 (麦克风)"
    echo "  ./run.sh list                          - 列出已注册声纹"
    echo "  ./run.sh identify [音频路径]           - 识别说话人"
    echo "  ./run.sh clean                         - 清理临时文件"
    echo ""
    echo "示例:"
    echo "  ./run.sh register '张三' samples/zhangsan.wav"
    echo "  ./run.sh meeting samples/meeting.wav"
    echo "========================================================"
}

# 检查参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND=$1

case $COMMAND in
    register)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "❌ 错误: 请提供姓名和音频路径"
            echo "用法: ./run.sh register [姓名] [音频路径]"
            exit 1
        fi
        python voiceprint.py register --name "$2" --audio "$3"
        ;;
        
    meeting)
        if [ -z "$2" ]; then
            echo "❌ 错误: 请提供会议音频路径"
            exit 1
        fi
        AUDIO_FILE=$2
        BASENAME=$(basename "$AUDIO_FILE")
        # 移除后缀
        FILENAME="${BASENAME%.*}"
        OUTPUT_FILE="${FILENAME}_notes.md"
        
        echo "🎙️  正在处理会议录音: $AUDIO_FILE"
        echo "📝 输出文件: $OUTPUT_FILE"
        
        python meeting.py --audio "$AUDIO_FILE" --output "$OUTPUT_FILE"
        ;;
        
    live)
        OUTPUT_FILE=${2:-"live_meeting.md"}
        echo "🔴 启动实时会议记录..."
        echo "📝 输出文件: $OUTPUT_FILE"
        python live.py --output "$OUTPUT_FILE"
        ;;
        
    list)
        python voiceprint.py list
        ;;
        
    identify)
        if [ -z "$2" ]; then
            echo "❌ 错误: 请提供音频路径"
            exit 1
        fi
        python voiceprint.py identify --audio "$2"
        ;;
        
    delete)
        if [ -z "$2" ]; then
            echo "❌ 错误: 请提供要删除的说话人姓名"
            echo "用法: ./run.sh delete [姓名]"
            exit 1
        fi
        python voiceprint.py delete --name "$2"
        ;;
        
    clean)
        echo "清理临时文件..."
        rm -f *.wav *.tmp
        echo "✅ 清理完成"
        ;;
        
    *)
        show_help
        ;;
esac

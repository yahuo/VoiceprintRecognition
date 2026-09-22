#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 固定使用项目 venv；不依赖调用目录，也不在缺少 venv 时回退系统 Python。
function run_python {
    local python="$PROJECT_DIR/venv/bin/python"
    if [ ! -x "$python" ]; then
        echo "❌ 缺少项目虚拟环境，请先创建 $PROJECT_DIR/venv 并安装 requirements.txt" >&2
        return 1
    fi
    PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$python" "$@"
}

# 打印帮助信息
function show_help {
    echo "========================================================"
    echo "  🎙️  MOSS / FunASR 声纹识别与会议记录工具箱"
    echo "========================================================"
    echo "用法:"
    echo "  ./run.sh register [姓名] [音频路径]    - 注册声纹"
    echo "  ./run.sh meeting [音频路径]            - 生成会议记录"
    echo "  ./run.sh live [输出文件]               - 🔴 实时会议记录 (麦克风)"
    echo "  ./run.sh list                          - 列出已注册声纹"
    echo "  ./run.sh identify [音频路径]           - 识别说话人"
    echo "  ./run.sh delete [声纹ID]               - 按 ID 删除声纹"
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
        if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
            echo "❌ 错误: 请提供姓名和音频路径"
            echo "用法: ./run.sh register [姓名] [音频路径]"
            exit 1
        fi
        run_python -m app.utils.voiceprint register --name "$2" --audio "$3"
        ;;
        
    meeting)
        if [ -z "${2:-}" ]; then
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
        
        run_python -m app.services.meeting --audio "$AUDIO_FILE" --output "$OUTPUT_FILE"
        ;;
        
    live)
        OUTPUT_FILE=${2:-"live_meeting.md"}
        echo "🔴 启动实时会议记录..."
        echo "📝 输出文件: $OUTPUT_FILE"
        run_python -m app.services.live --output "$OUTPUT_FILE"
        ;;
        
    list)
        run_python -m app.utils.voiceprint list
        ;;
        
    identify)
        if [ -z "${2:-}" ]; then
            echo "❌ 错误: 请提供音频路径"
            exit 1
        fi
        run_python -m app.utils.voiceprint identify --audio "$2"
        ;;
        
    delete)
        if [ -z "${2:-}" ]; then
            echo "❌ 错误: 请提供要删除的声纹 ID（不是姓名）"
            echo "用法: ./run.sh delete [声纹ID]"
            exit 1
        fi
        run_python -m app.utils.voiceprint delete --id "$2"
        ;;

    help|-h|--help)
        show_help
        ;;

    *)
        echo "❌ 未知命令: $COMMAND" >&2
        show_help
        exit 2
        ;;
esac

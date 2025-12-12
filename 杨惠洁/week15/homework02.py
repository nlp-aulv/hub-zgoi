import subprocess
import os
import shutil
import glob  # 导入 glob 库


def parse_pdf_via_cli(pdf_path: str, output_dir: str = "mineru_cli_temp"):
    """通过调用命令行接口来解析 PDF，并返回 Markdown 文本。"""

    absolute_pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(absolute_pdf_path):
        return f"错误：文件不存在于路径 '{absolute_pdf_path}'"

    try:
        # 1. 准备输出目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        #  新增：设置环境变量使用国内镜像
        env = os.environ.copy()  # 复制当前环境变量
        env['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置国内镜像
        # 可选：设置缓存目录
        # env['HF_HOME'] = 'D:/huggingface_cache'

        # 2. 构造命令行命令
        command = [
            "mineru",
            "-p", absolute_pdf_path,
            "-o", output_dir,
            "--to", "markdown"
        ]

        # 3. 执行命令（传入环境变量）
        print(f"执行命令: {' '.join(command)}")
        print(f"使用HF镜像: {env.get('HF_ENDPOINT', '默认')}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            timeout=300,  # 增加超时时间
            env=env  #  传入修改后的环境变量
        )

        # 打印输出信息
        if result.stdout:
            print(f"MinerU STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"MinerU STDERR (可能有警告/提示):\n{result.stderr}")

        # 4. 查找生成的MD文件
        search_pattern = os.path.join(output_dir, "*.md")
        list_of_files = glob.glob(search_pattern)

        if not list_of_files:
            # 再次打印输出目录内容，确认是否生成了其他格式文件
            print(f"输出目录内容: {os.listdir(output_dir)}")
            return f"错误：命令行执行成功，但未找到任何 .md 文件在 {output_dir} 中。"

        # 找到最新修改的文件
        latest_file = max(list_of_files, key=os.path.getmtime)
        markdown_file = latest_file
        print(f" 成功找到输出文件: {markdown_file}")

        with open(markdown_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        return text_content

    except subprocess.CalledProcessError as e:
        return f"命令行执行失败 (返回码: {e.returncode}):\n{e.stderr}\nSTDOUT:\n{e.stdout}"
    except subprocess.TimeoutExpired:
        return "命令行执行超时 (超过 180 秒)。"
    except Exception as e:
        return f"解析过程中发生错误: {e}"


# --- 示例使用 ---

# 确保这里的路径是准确的，并且文件确实存在
PDF_FILE_PATH = "D:/networkStudy/Week15/Week15/模型论文/2507-PaddleOCR 3.0.pdf"
extracted_content = parse_pdf_via_cli(PDF_FILE_PATH)
print(extracted_content)
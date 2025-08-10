import os
from openai import OpenAI
import time
from tqdm import tqdm

# 配置API客户端
client = OpenAI(
    api_key="sk-96f5e4fd40c247d49066ae70ffec4b45",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def extract_content(text):
    """调用LLM API提取正文"""
    try:
        completion = client.chat.completions.create(
            model="qwen-plus-latest",
            messages=[
                {"role": "system", "content": "你是一个文本处理助手。请从用户提供的网页爬虫文本中提取出正文内容，只输出正文部分，不要任何其他内容、解释或格式。"},
                {"role": "user", "content": f"请提取以下文本中的正文部分：\n\n{text}"},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用失败: {e}")
        return None

def process_files():
    source_dir = "british_pdf_2024_txt"
    target_dir = "british_pdf_2024_txt_clean"
    
    # 创建目标根目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    processed_count = 0
    failed_count = 0
    
    # 收集所有需要处理的文件
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                source_path = os.path.join(root, file)
                all_files.append(source_path)
    
    # 使用tqdm显示进度
    for source_path in tqdm(all_files, desc="处理文件"):
        # 构建目标路径
        relative_path = os.path.relpath(source_path, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        target_folder = os.path.dirname(target_path)
        
        # 创建目标目录
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        try:
            # 读取源文件
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 长度过滤
            if len(content) > 1024*32:
                extracted_content = content
            else:
            # 提取正文
                extracted_content = extract_content(content)
            
            if extracted_content:
                # 保存到目标文件
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_content)
                processed_count += 1
                tqdm.write(f"✓ 成功处理: {relative_path}")
            else:
                failed_count += 1
                tqdm.write(f"✗ 处理失败: {relative_path}")
            
            # 添加延时避免API频率限制
            # time.sleep(0.5)
            
        except Exception as e:
            failed_count += 1
            tqdm.write(f"✗ 文件处理错误 {source_path}: {e}")
    
    print(f"\n处理完成！成功: {processed_count}, 失败: {failed_count}")

if __name__ == "__main__":
    process_files()

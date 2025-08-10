import PyPDF2
import os
import shutil


def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def pdf_to_txt(pdf_path, output_dir):
    """
    将单个PDF文件转换为TXT文本并保存到指定目录

    参数:
        pdf_path (str): PDF文件的路径
        output_dir (str): 输出目录
    """
    try:
        # 获取PDF文件名（不含扩展名）
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_filename = f"{pdf_filename}.txt"
        txt_path = os.path.join(output_dir, txt_filename)

        # 打开PDF文件
        with open(pdf_path, 'rb') as pdf_file:
            # 创建PDF阅读器对象
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # 获取PDF的总页数
            num_pages = len(pdf_reader.pages)

            # 提取文本内容
            text_content = []
            for page_num in range(num_pages):
                # 获取第page_num页
                page = pdf_reader.pages[page_num]

                # 提取页面文本
                page_text = page.extract_text()

                if page_text:
                    # text_content.append(f"--- 第 {page_num + 1} 页 ---")
                    text_content.append(page_text)
                else:
                    print(f"警告: {pdf_path} 第 {page_num + 1} 页没有可提取的文本")

            # 将提取的文本写入TXT文件
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write('\n\n'.join(text_content))

            print(f"已处理: {pdf_path} -> {txt_path}")

    except Exception as e:
        print(f"处理 {pdf_path} 时发生错误: {str(e)}")


def process_pdf_files_recursively(input_dir, output_root_dir):
    """
    递归处理目录中的所有PDF文件

    参数:
        input_dir (str): 输入根目录
        output_root_dir (str): 输出根目录
    """
    # 确保输出根目录存在
    ensure_directory_exists(output_root_dir)

    # 遍历输入目录中的所有项目
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)

        # 如果是目录，则递归处理
        if os.path.isdir(item_path):
            # 构建对应的输出目录
            relative_path = os.path.relpath(item_path, input_dir)
            output_dir = os.path.join(output_root_dir, relative_path)
            ensure_directory_exists(output_dir)

            # 递归处理子目录
            process_pdf_files_recursively(item_path, output_root_dir)

        # 如果是PDF文件，则进行转换
        elif os.path.isfile(item_path) and item.lower().endswith('.pdf'):
            # 确定该文件对应的输出目录
            relative_dir = os.path.relpath(os.path.dirname(item_path), input_dir)
            output_dir = os.path.join(output_root_dir, relative_dir)
            ensure_directory_exists(output_dir)

            # 转换PDF为TXT
            pdf_to_txt(item_path, output_dir)

import os


def is_empty_file(file_path):
    """检查文件是否为空"""
    return os.path.getsize(file_path) == 0


def delete_empty_txt_files(directory):
    """删除目录及其子目录中所有空的TXT文件"""
    # 先处理子目录，再处理当前目录（确保先删除文件再检查空文件夹）
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isdir(item_path):
            # 递归处理子目录
            delete_empty_txt_files(item_path)
        elif os.path.isfile(item_path) and item.lower().endswith('.txt'):
            # 检查并删除空TXT文件
            if is_empty_file(item_path):
                try:
                    os.remove(item_path)
                    print(f"已删除空TXT文件: {item_path}")
                except Exception as e:
                    print(f"删除文件 {item_path} 失败: {str(e)}")


def delete_empty_folders(directory, delete_root=False):
    """
    删除目录及其子目录中所有空文件夹

    参数:
        directory (str): 要处理的目录
        delete_root (bool): 是否删除根目录（即使它为空）
    """
    # 检查目录是否存在
    if not os.path.isdir(directory):
        return False

    # 先处理子目录
    has_files = False
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # 递归处理子目录，如果子目录被删除则返回True
            if delete_empty_folders(item_path, True):
                continue
        # 如果有任何文件或非空文件夹，则标记当前目录不为空
        has_files = True

    # 如果目录为空且允许删除，则删除它
    if not has_files and delete_root:
        try:
            os.rmdir(directory)
            print(f"已删除空文件夹: {directory}")
            return True
        except Exception as e:
            print(f"删除文件夹 {directory} 失败: {str(e)}")
            return False

    return False


def clean_empty(target_dir):
    """清理target_dir目录中的空文件和空文件夹"""
    

    # 检查目标目录是否存在
    if not os.path.exists(target_dir):
        print(f"目录 '{target_dir}' 不存在，无需清理")
        return

    print(f"开始清理目录: {target_dir}")

    # 第一步：删除所有空TXT文件
    delete_empty_txt_files(target_dir)

    # 第二步：删除所有空文件夹（不删除根目录，除非它也为空）
    delete_empty_folders(target_dir, delete_root=False)

    print("清理完成")

if __name__ == "__main__":
    # 输入目录和输出目录
    input_directory = "british_pdf_2024"
    output_directory = "british_pdf_2024_txt"

    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录 '{input_directory}' 不存在")
    else:
        print(f"开始处理目录: {input_directory}")
        print(f"输出将保存到: {output_directory}")
        process_pdf_files_recursively(input_directory, output_directory)

        clean_empty(output_directory)
        print("处理完成")

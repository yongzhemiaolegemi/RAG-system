import pdfplumber

def extract_text_with_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 使用示例
pdf_text = extract_text_with_pdfplumber('example.pdf')
print(pdf_text)
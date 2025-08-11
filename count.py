import json 

working_dir= 'british_database'



 
for k in ['entities', 'relationships', 'chunks']:
    file_path = f"{working_dir}/vdb_{k}.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"# {k}: {len(data['data'])}")


kv_store_doc_status = f"{working_dir}/kv_store_doc_status.json"
with open(kv_store_doc_status, 'r', encoding='utf-8') as file:
    data = json.load(file)

print(f"# doc: {len(data)}") 


kv_store_text_chunks = f"{working_dir}/kv_store_text_chunks.json"


def calculate_token_cnt(data):
    total = 0
    
    # 检查数据是否为字典类型
    if not isinstance(data, dict):
        print("数据不是字典类型，无法计算")
        return 0
    
    # 遍历字典中的每个值（即每个文档对象）
    for doc in data.values(): 
        if isinstance(doc, dict) and "tokens" in doc: 
            if isinstance(doc["tokens"], (int, float)):
                total += doc["tokens"]
            else:
                print(f"警告：tokens不是数字类型，已忽略")
        else:
            print(f"警告：文档对象不包含tokens字段，已忽略")
    
    return total
with open(kv_store_text_chunks, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(f"# tokens: {calculate_token_cnt(data)}") 

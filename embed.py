from zhipuai import ZhipuAI
import numpy as np

client = ZhipuAI(api_key="a187b05acc4b472187d035f3c4fd1112.GIgX9BJBdhMV28ju") 
response = client.embeddings.create(
    model="embedding-3", #填写需要调用的模型编码
     input=[
        "美食非常美味，服务员也很友好。",
        "这部电影既刺激又令人兴奋。",
        "阅读书籍是扩展知识的好方法。"
    ],
)
embeddings = np.array([item.embedding for item in response.data])
print(embeddings.shape)  # 输出: (3, 2048)
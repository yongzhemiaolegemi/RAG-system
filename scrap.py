import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
from collections import deque
import requests
from filter import filter_by_llm
from utils import config
import os

def get_webpage_text(url):
    try:
        # 发送HTTP请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 移除不需要的元素（如脚本、样式等）
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 获取文本内容
        text = soup.get_text(separator='\n', strip=True)
        
        return text
    
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except Exception as e:
        print(f"处理错误: {e}")
        return None

async def get_webpage_text(url, session, condition):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                print(f"请求失败 {url}: HTTP {response.status}")
                return None
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            txt_content =  soup.get_text(separator='\n', strip=True)
            judgement, reason = filter_by_llm(txt_content, condition)
            print(f"判断结果 {url}: {judgement}, 理由: {reason}")
            if judgement:

                # save to config.webscrap_base_dir+condition
                file_path = f"{config().webscrap_base_dir}/{condition}/{url.replace('http://', '').replace('https://', '').replace('/', '_')}.txt"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(txt_content)
                
            return txt_content

    
    except Exception as e:
        print(f"处理错误 {url}: {str(e)}")
        return None

async def crawl_website_async(base_url,condition, max_pages=5, max_concurrent=10):
    visited = set()
    to_visit = deque([base_url])
    results = {}
    
    async with aiohttp.ClientSession(headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }) as session:
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_page(url):
            async with semaphore:
                if url in visited or len(visited) >= max_pages:
                    return set()
                
                visited.add(url)
                print(f"正在处理: {url}")
                
                try:
                    # 并行执行：同时获取文本和提取链接
                    text, links = await asyncio.gather(
                        get_webpage_text(url, session, condition),
                        extract_links(url, session)
                    )
                    
                    if text:
                        results[url] = text
                    return links
                
                except Exception as e:
                    print(f"处理失败 {url}: {str(e)}")
                    return set()
        
        async def extract_links(url, session):
            try:
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    new_links = set()
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(base_url, href)
                        
                        if (urlparse(full_url).netloc == urlparse(base_url).netloc and 
                            full_url not in visited and 
                            full_url not in to_visit):
                            new_links.add(full_url)
                    
                    return new_links
            except:
                return set()
        
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.popleft()
            new_links = await process_page(current_url)
            
            for link in new_links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)
    
    # form results as "<url>: <text>"
    text_results = {f"{url}:\n {text}\n\n" for url, text in results.items() if text}
    return text_results


# 示例用法
async def main():
    base_url = "https://www.afro.who.int/health-topics/communicable-diseases"
    results = await crawl_website_async(base_url, condition="网页内容范围是2023年9月至今",max_pages=20, max_concurrent=5)
    

if __name__ == "__main__":
    asyncio.run(main())
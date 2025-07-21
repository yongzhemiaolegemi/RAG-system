import requests
from bs4 import BeautifulSoup

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

# 示例使用
url_list = [
    'https://africacdc.org/news-item/africa-cdc-and-mastercard-foundation-launch-phase-2-of-the-saving-lives-and-livelihoods-sll-initiative-to-strengthen-africas-public-health-systems/',
    'https://africacdc.org/news-item/africa-cdc-launches-initiatives-to-advance-molecular-diagnostics-and-genomic-surveillance-in-africa/',
    'https://www.afro.who.int/news/new-framework-launched-eliminate-visceral-leishmaniasis-eastern-africa',
    'https://www.afro.who.int/news/african-health-leaders-begin-work-roadmap-reshape-global-health-financing-continent',
    'https://africacdc.org/news-item/africa-cdc-and-africa-public-health-foundation-forge-strategic-partnership-to-strengthen-health-systems/',
    'https://www.afro.who.int/news/african-regions-first-ever-health-workforce-investment-charter-launched',
    'https://africacdc.org/news-item/communique-united-in-the-fight-against-mpox-in-africa-high-level-emergency-regional-meeting/',
    'https://africacdc.org/news-item/african-researchers-propose-mpox-research-group/',
    'https://www.afro.who.int/news/forging-resilience-who-afro-establishes-independent-expert-body-bolster-africas-health',
    'https://www.afro.who.int/news/who-africa-welcomes-gavis-commitment-africa-vaccine-manufacture-immunization-and-pandemic',
    'https://www.afro.who.int/news/nearly-10-000-children-vaccinated-malaria-vaccine-rollout-africa-expands',
    'https://africacdc.org/news-item/enhancing-vaccine-storage-efficiencies-in-africa/',
    'https://africacdc.org/news-item/addressing-regulatory-challenges-to-advance-local-manufacturing-in-africa/',
    'https://africacdc.org/news-item/african-health-ministers-commit-to-purchasing-locally-made-vaccines/',
    "https://africacdc.org/news-item/african-health-ministers-commit-to-purchasing-locally-made-vaccines/",
    "https://africacdc.org/news-item/the-african-vaccine-manufacturing-accelerator-is-a-boon-for-the-continent/",
    "https://www.afro.who.int/news/senegal-who-launch-regional-emergency-hub-bolster-africas-response-health-crises",
    "https://afhro.org/africa-cdc-and-who-launch-joint-emergency-preparedness-and-response-action-plan-jeap-to-strengthen-health-systems-and-combat-disease-outbreaks-in-africa/",
    "https://africacdc.org/news-item/partnership-for-climate-and-disaster-response-seeks-1-billion-usd-who-and-africa-cdc-strengthen-efforts-to-tackle-health-emergencies-in-africa/",
    "https://www.afro.who.int/news/who-africa-bill-and-melinda-gates-foundation-pursue-collaboration-leverage-data-analytics",
    "https://africacdc.org/news-item/africa-cdc-and-unicef-expand-partnership-to-strengthen-health-systems-and-immunization-of-children-in-africa/",
    "https://africacdc.org/news-item/africa-cdc-and-cepi-deepen-partnership-to-fortify-african-preparedness-against-disease-outbreaks/",
    "https://africacdc.org/news-item/press-statement-africa-cdc-committed-to-supporting-the-replenishment-of-gavi-funds-in-paris/",
    "https://africacdc.org/news-item/africa-cdc-and-eu-join-efforts-to-improve-equitable-access-to-health-products-and-local-manufacturing-for-africa/",
    "https://africacdc.org/news-item/high-level-event-kicks-off-expansion-of-strategic-eu-au-partnership-pledging-joint-commitments-to-strengthen-global-health-and-african-health-sovereignty/",
    "https://africacdc.org/news-item/africa-cdc-and-france-sign-memorandum-of-understanding-to-strengthen-public-health-systems-in-africa/",
    "https://africacdc.org/news-item/africa-cdc-who-welcome-uks-support-to-address-health-challenges/",
    "https://africacdc.org/news-item/joint-statement-on-the-africa-cdc-and-united-states-new-joint-action-plan/",
    "https://africacdc.org/news-item/mpox-outbreaks-in-africa-constitute-a-public-health-emergency-of-continental-security/",
    "https://africacdc.org/news-item/african-health-ministers-commit-to-purchasing-locally-made-vaccines/",
    "https://www.afro.who.int/news/african-regions-first-ever-health-workforce-investment-charter-launched",
    "https://www.afro.who.int/news/pioneering-charter-drive-investment-africas-health-workforce",
    "https://africacdc.org/news-item/antimicrobial-resistance-is-a-greater-threat-than-hiv-aids-tb-and-malaria-says-new-report/"

    
]

import os
folder_dir = 'files'
if not os.path.exists(folder_dir):
    os.makedirs(folder_dir)


for url in url_list:
    text = get_webpage_text(url)
    print(f"URL: {url}\nText Length: {len(text)}\n")
    if text:
        # remove last /
        if url.endswith('/'):
            url = url[:-1]
        filename = os.path.join(folder_dir, url.split('/')[-1] + '.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
    

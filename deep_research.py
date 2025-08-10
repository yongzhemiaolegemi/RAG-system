import json
from typing import Tuple
import uuid # 确保 uuid 被导入，用于流式处理中的 tool_call id 生成
import os

from openai import OpenAIError # 用于更精确地捕获 OpenAI 相关异常
import requests
from openai import OpenAI
from utils import config
import re
import json
from collections import defaultdict
import time
import math
from time import sleep
from requests.exceptions import ConnectionError as RequestsConnectionError
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)

INITIAL_QUESTIONS_COUNT = 15 # 初始生成的问题数量
MAX_ADDITIONAL_QUESTIONS = 5 # 最多允许额外增加的问题数量
SOURCE_DOCUMENTS_DIR = "./africa2024_raw_files/" # 源文档所在的目录
POLISH = False
CONNECTION_TEST_BEFORE_STAGE_CONCLUSION = True # 总结阶段之前进行一次连通性测试
QUERY_PROMPT = "给出参考文献时，请忠实给出原文档名，不要翻译成中文，也不要把relation等误当作文档名。"
# TOPIC = "需要一个标题为：“非洲生物安全态势研判”的报告。字数大约在2000~3000字。分为三个章节：1.非洲生物安全态势 2.非洲生物安全风险点 3.对我国(中国)应对风险及加强非洲国际合作的建议"
# TOPIC = "需要一个标题为：“非洲生物安全态势研判”的报告。几百字即可"
TOPIC = '''
撰写一篇智库研究报告，标题为：
《英国生物安全态势研判》
本报告面向政策研究机构内部参考，供政府部门领导研判决策使用。全文建议控制在 6000 字左右，内容要求如下：
________________________________________
报告结构与内容指引
1. 英国生物安全总体态势
•	概述英国当前生物安全形势；
•	引用 2024年 9 月以来的最新数据或正式报告支撑分析。
2. 当前面临的主要生物安全风险与深度分析
•	梳理主要风险类型，并提供实际案例或数据支持；
•	强调风险的系统性与持续性，指出短期内难以缓解的问题；
•	重要的小概率事件
3. 中国应对策略与中英合作建议
•	建议应紧扣前述风险与问题提出；
•	所有建议应具备一定可行性与针对性；
•	可结合中国相关政策立场提出合作方向建议，避免泛泛而谈。
________________________________________
写作风格与技术规范
•	使用政策研究风格语言，理性、专业、逻辑清晰；
•	引用权威数据与正式文件时请标注来源与时间；
•	缩略词请在首次出现时注明全称；
•	报告应避免空泛，突出问题导向与政策建议的适用性。
________________________________________
输出目标：
生成一篇逻辑清晰、数据充分、问题导向明确、建议具有可操作性的智库研究报告，可供政府部门领导研判使用。
内容上请勿重复冗余。
请在合适的地方插入至少一个表格，用于总结时间线、横向对比、纵向对比等用途。

'''

def quick_connectivity_test(url, key, model):
    """
    快速发一个非常短的请求来测试连通性与认证（诊断用）。
    """
    try:
        tmp_client = OpenAI(api_key=key, base_url=url)
        messages = [
            {"role": "system", "content": "You are a diagnostic bot."},
            {"role": "user", "content": "Say OK in one word."}
        ]
        resp = tmp_client.chat.completions.create(model=model, messages=messages, timeout=30)
        print("短请求成功，resp type:", type(resp), "，长度:", len(str(resp)))
    except Exception as e:
        import traceback
        print("短请求失败：", repr(e))
        traceback.print_exc()


def post_to_openai_api(messages, model, url, key, stream=False, collect_stream=True, tools=None):
    try:
        client = OpenAI(
            api_key=key,
            base_url=url
        )
        
        # 构建API调用参数
        api_params = {
            "model": model,
            "messages": messages,
            # "extra_body": {"enable_thinking": True},
            "stream": stream,
            "timeout": 600
        }
        
        # 如果提供了工具，添加到API调用中
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        print(f"调用API: {url}")
        print(f"模型: {model}")
        print(f"消息数量: {len(messages)}")
        
        # 计算消息体大小（字符数）
        total_chars = sum(len(m.get("content") or "") for m in messages)
        total_tokens_est = math.ceil(total_chars / 4)  # 粗略估算：1 token ≈4 chars
        print(f"消息体字符数: {total_chars}, 估算 token: {total_tokens_est}")
        max_retries = 3
        retry_delay = 1
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(**api_params)
                break
            except Exception as api_error:
                # 如果是连接相关错误，尝试重试（指数退避）
                is_conn_err = isinstance(api_error, RequestsConnectionError) or "Connection" in str(api_error)
                print(f"API 调用第 {attempt} 次失败: {api_error}")
                if attempt == max_retries or not is_conn_err:
                    # 最后一次或非连接类错误，抛出
                    raise api_error
                else:
                    sleep_time = retry_delay * (2 ** (attempt - 1))
                    print(f"检测到连接错误，{sleep_time}s 后重试...")
                    sleep(sleep_time)
        
    except Exception as api_error:
        import traceback, sys
        print("API 调用发生异常，类型:", type(api_error).__name__)
        print("异常 repr:", repr(api_error))
        print("完整 traceback:")
        traceback.print_exc()
        print(f"API URL: {url}")
        print(f"API Key 前 20 字符: {key[:20]}...")
        # 简单的 URL 可达性测试（仅做诊断，不作为最终请求）
        try:
            import requests as _requests
            r = _requests.get(url, timeout=5, verify=True)
            print("URL GET 返回状态:", r.status_code)
            print("返回内容（前200字）:", r.text[:200])
        except Exception as e:
            print("URL 可达性测试失败:", repr(e))
        # 重新抛出原始异常以便上层看到
        raise
    
    if stream:
        if collect_stream:
            # 收集所有流式数据并拼接
            collected_content = ""
            tool_calls = []
            full_response = None
            
            for chunk in completion:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # 收集内容
                    if hasattr(delta, 'content') and delta.content:
                        collected_content += delta.content
                    
                    # 收集工具调用
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.index >= len(tool_calls):
                                tool_calls.extend([None] * (tool_call.index + 1 - len(tool_calls)))
                            
                            if tool_calls[tool_call.index] is None:
                                tool_calls[tool_call.index] = {
                                    "id": tool_call.id or f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name or "",
                                        "arguments": tool_call.function.arguments or ""
                                    }
                                }
                            else:
                                # 追加参数
                                if tool_call.function.arguments:
                                    tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                    
                    # 保存最后一个chunk作为模板
                    full_response = chunk.model_dump()
            
            # 构建完整响应
            if full_response:
                choice_data = {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": collected_content if collected_content else None,
                    },
                    "finish_reason": "stop"
                }
                
                # 如果有工具调用，添加到响应中
                if tool_calls and any(tc for tc in tool_calls):
                    choice_data["message"]["tool_calls"] = [tc for tc in tool_calls if tc]
                    choice_data["finish_reason"] = "tool_calls"
                
                full_response["choices"] = [choice_data]
                
            return full_response or {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant", 
                        "content": collected_content if collected_content else None
                    },
                    "finish_reason": "stop"
                }]
            }
        else:
            # 返回生成器函数用于流式响应
            def generate():
                for chunk in completion:
                    chunk_data = chunk.model_dump()
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
            return generate()
    else:
        # 非流式响应，直接返回字典
        return completion.model_dump()

def send_query_to_RAG_server(query: str, mode: str = "hybrid", url: str = config().lightrag_service_url,user_prompt='') -> Tuple[str, str] :
    """向RAG服务器发送查询"""
    # POST to url
    # {
    #     "message": "你的查询问题",
    #     "mode": "查询模式"
    # }
    data = {
        "message": query,
        "mode": mode,
        'user_prompt':user_prompt,
        "deep_research": True  # 标明这个query请求是在进行deep research的过程中发出的，这样的话rag系统在生成内容时，将不会在结尾处完整列出参考文献
    }
    response = requests.post(url, json=data)
    return response.json()["result"], response.json()["log_file_path"]


class ResearchAgent:

    def __init__(self, topic, model_name="gpt-4.1-ca", initial_questions=INITIAL_QUESTIONS_COUNT, max_extra_questions=MAX_ADDITIONAL_QUESTIONS):

        self.state = {
            "original_topic": topic,
            "question_list": [],
            "completed_qa_pairs": [],
            "research_log": [], # 可选：记录研究过程
            "added_questions_count": 0 # 记录额外增加的问题数量
        }
        self.model = model_name
        self.initial_questions_count = initial_questions
        self.max_additional_questions = max_extra_questions

    def _call_llm(self, system_prompt, user_prompt, tools=None, model_name=None, url=None, key=None, stream=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = post_to_openai_api(messages, model_name or self.model, url, key, stream=stream, tools=tools)
            
            content = response['choices'][0]['message'].get('content')
            if content:
                return content.strip()
            else:
                print("警告: LLM响应中没有内容。")
                return ""
        except (OpenAIError, KeyError, IndexError) as e:
            print(f"调用LLM时出错: {e}")
            return ""


    def _parse_json_from_llm_response(self, response_text):
        if not response_text:
            return []
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start_marker = "```json"
            end_marker = "```"
            start_index = response_text.find(start_marker)
            end_index = response_text.rfind(end_marker)
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_str = response_text[start_index + len(start_marker):end_index].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"无法解析JSON代码块: {json_str}")
                    return []
            else:
                print(f"LLM响应格式不正确，无法找到JSON数组: {response_text}")
                return []

    def plan_research(self):
        print("--- 阶段一：规划研究 ---")
        system_prompt = (
            "你是一位资深行业研究员。你的任务是将复杂的研究主题分解为一系列具体、清晰、可独立回答的问题。"
            "这些问题将用于从知识库中检索信息。"
        )
        user_prompt = (
            f"研究主题是：{self.state['original_topic']}\n\n"
            f"请设计一个全面的研究大纲，并将其分解为 {self.initial_questions_count} 个核心问题。"
            "问题应覆盖主题的各个方面，逻辑清晰，表述明确。"
            "请严格按照以下格式返回：一个包含问题的字符串list，不要有其他任何文字或解释。"
            "例如: [\"问题1？\", \"问题2？\", ...]"
        )

        response_text = self._call_llm(system_prompt, user_prompt, model_name=config().dr_query_model, url=config().dr_query_model_url, key=config().dr_query_model_key)
        question_list = self._parse_json_from_llm_response(response_text)

        if not question_list:
            print("警告：未能从LLM获取有效的初始问题列表。研究可能无法继续。")
            exit()
            question_list = [f"关于{self.state['original_topic']}的关键问题是什么？"]

        self.state['question_list'] = question_list
        print(f"已生成初始问题列表 ({len(question_list)} 个问题)。")

    def execute_and_reflect(self):
        print("\n--- 阶段二：迭代研究与反思 ---")
        iteration = 1
        while self.state['question_list']:
            print(f"\n--- 迭代 {iteration} ---")
            current_question = self.state['question_list'].pop(0) # 取出第一个问题
            if not isinstance(current_question, str):
                current_question = json.dumps(current_question)
            print(f"正在研究问题: {current_question}")

            try:
                rag_answer, log_file_path = send_query_to_RAG_server(current_question, mode = "hybrid_dc")
                print(f"RAG回答: {rag_answer[:100]}...") # 打印部分回答作为反馈
            except Exception as e:
                print(f"调用RAG服务时出错: {e}")
                rag_answer = "未能从知识库获取相关信息。" # 默认回答

            self.state['completed_qa_pairs'].append({
                "question": current_question,
                "answer": rag_answer,
                "log_path": log_file_path
            })

            system_prompt = (
                "你是一位研究策略师。你的任务是根据已收集的信息，动态优化后续的研究计划。"
                "你需要审视当前的问题列表，并根据新获得的知识对其进行修改、添加或删除。"
            )
            
            formatted_qa = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}\n" for pair in self.state['completed_qa_pairs']])
            formatted_todo = "\n".join([f"{i+1}. {q}" for i, q in enumerate(self.state['question_list'])])

            remaining_allowed = self.max_additional_questions - self.state['added_questions_count']
            user_prompt = (
                f"原始研究主题: {self.state['original_topic']}\n\n"
                f"我们已经完成的研究问答:\n{formatted_qa}\n\n"
                f"我们接下来计划研究的问题列表:\n{formatted_todo}\n\n"
                "请基于以上信息，对剩余的问题列表进行优化。\n"
                "你可以：\n"
                "1. 修改现有问题，使其更深入或更准确。\n"
                "2. 添加由新信息启发的新问题。\n"
                "3. 删除已经解答或不再相关的问题。\n"
                f"注意：你最多还可以添加 {remaining_allowed} 个新问题。\n"
                "确保最终问题列表逻辑连贯、无重复，并能支撑一份完整的报告。\n"
                "请严格按照以下格式返回：一个包含更新后问题的JSON数组，不要有其他任何文字或解释。"
                "例如: [\"修改后的问题1？\", \"新问题？\", ...]"
            )

            response_text = self._call_llm(system_prompt, user_prompt, model_name=config().dr_query_model, url=config().dr_query_model_url, key=config().dr_query_model_key)
            updated_question_list = self._parse_json_from_llm_response(response_text)

            if updated_question_list is not None: # 即使是空列表也是有效的更新
                # 限制可以添加的新问题数量
                questions_before_reflection = len(self.state['question_list'])
                newly_added_count = len(updated_question_list) - questions_before_reflection

                if newly_added_count > 0:
                    if self.state['added_questions_count'] >= self.max_additional_questions:
                        print(f"警告：已达到额外问题上限({self.max_additional_questions})，新问题将被忽略。")
                        updated_question_list = updated_question_list[:questions_before_reflection]
                    else:
                        allowed_new = self.max_additional_questions - self.state['added_questions_count']
                        if newly_added_count > allowed_new:
                            print(f"警告：本次新增问题过多，只保留 {allowed_new} 个。")
                            updated_question_list = updated_question_list[:questions_before_reflection + allowed_new]
                            self.state['added_questions_count'] = self.max_additional_questions
                        else:
                            self.state['added_questions_count'] += newly_added_count
                
                print(f"反思后，问题列表已更新 (剩余 {len(updated_question_list)} 个问题)。")
                self.state['question_list'] = updated_question_list
            else:
                print("警告：反思阶段未能获取有效的更新后问题列表，将保留当前列表。")
            
            iteration += 1

        print("\n--- 阶段二完成：所有问题均已研究完毕。---")

    def synthesize_report(self):
        print("\n--- 阶段三：综合生成报告 ---")
        if not self.state['completed_qa_pairs']:
            print("没有收集到任何信息，无法生成报告。")
            return "无法生成报告：未收集到任何信息。"

        system_prompt = (
            "你是一位专业的报告撰写专家。你的任务是根据提供的原始主题和一系列问答对，撰写一份结构清晰、逻辑严谨、内容详实的最终研究报告。"
        )
        
        # 格式化所有QA对作为报告素材
        formatted_qa = "\n\n".join([
            f"**问题 {i+1}:** {pair['question']}\n\n**回答:** {pair['answer']}" 
            for i, pair in enumerate(self.state['completed_qa_pairs'])
        ])

        user_prompt = (
            f"原始研究主题: {self.state['original_topic']}\n\n"
            f"根据以下研究资料撰写报告：\n\n{formatted_qa}\n\n"
            "要求：\n"
            "1. 报告应包含引言、主体和结论。\n"
            "2. 主体部分应逻辑清晰地组织信息，可以分章节。\n"
            "3. 语言流畅、专业。\n"
            "4. 直接输出报告正文，无需额外说明。\n"
            "5. 每个大章节下面建议布置2~4个小节，视信息丰富程度而定。\n"
            "6. 每个小节的主体是1~2段内容丰富、语言连续的段落，请不要大量分点。每个小节请不要只写一句话就草草了事。请至少写三句话。请记住，你是在写一个内容丰富的报告，而不是在机械地列重点。\n"
            "7. 尽量多出现一些和时间有关的叙述，用以提升报告的时效性。例如xxxx年xx月这种。\n"
            "8. 在句子后标注引用来自哪个问题的回复，这些标注将会被程序识别然后进行后处理。如，如果是来自问题1的回复中的[DC #16]，那么就在句子后面标注[Q #1, DC #16]。如果有多个来源，可以写成[Q #1, DC #4-6 #15 #17]这种。如果一个句子涉及到两个及以上的问题的回复，例如同时涉及到了第一个问题的回复和第二个问题的回复，那么可以写成[Q #1, DC #3; Q #2, DC #3 #5]这种。Q的后面必须带有DC，禁止出现[Q #x]、[Q #5, DC 无]这种只有问题编号却无实际内容的标注。禁止出现[DC # 4]、[Q #8; DC #3 #7 #10]这种只有DC却无问题编号的标注。禁止出现[Q #7, DC 多][Q #5, DC 无]这种无法被解析的标注。不要在数字编号之间添加逗号，例如[Q #1, DC #1, #6]这种就是错误的，应该写成[Q #1, DC #1 #6]。\n"
            "9. 遇到RAG系统无法回答的问题时，请避开相关议题，避免引入错误信息。因为RAG系统所包含的知识并非十全十美，不能因为RAG系统没有查询到就下一些例如**缺乏透明度**之类的论断，这是不科学的。直接避开就好。但是与此同时你也不要在正文里承认RAG系统的不足，这会影响报告的质量。"
        )

        final_report = self._call_llm(system_prompt, user_prompt, model_name=config().dr_conclusion_model, url=config().dr_conclusion_model_url, key=config().dr_conclusion_model_key, stream=False)
        print("\n--- 阶段三完成：报告已生成。---")
        return final_report

    def polish_report(self, raw_report: str) -> str:
        system_prompt = (
            "你是中文政策报告的高级润色专家。"
            "在不改变事实、不改动引用标注、不移动标注位置的前提下，优化行文逻辑、术语一致性与可读性，修正轻微语病，消除冗余。"
        )
        user_prompt = (
            "润色要求：\n"
            "1) 严格保留并原样输出所有形如 [Q #... , ...] 的方括号引用标注，禁止改动其中任何字符、空格或顺序，禁止移动其在句中的位置；"
            "禁止新增、删除、合并或拆分任一引用标注。\n"
            "2) 不改变段落与换行结构；不引入新的引用标注或编号。\n"
            "3) 仅润色正文语言，不增删事实信息；输出仅为润色后的全文，不要添加任何解释或额外内容。\n\n"
            "待润色文本：\n"
            f"{raw_report}"
        )
        return self._call_llm(system_prompt, user_prompt, model_name=config().dr_polish_model, url=config().dr_polish_model_url, key=config().dr_polish_model_key)


    def _post_process_report_and_add_references(self, raw_report, references_map):
        """
        后处理报告：将引用标记转换为数字序号，并生成参考文献列表
        """
        print("\n--- 开始后处理报告：转换引用格式和生成参考文献 ---")
        
        # 初始化
        # references_map = {}  # 文档名 -> 引用序号
        citation_counter = 1
        question_log_data = {}  # 问题序号 -> 日志JSON数据
        
        # 预加载所有日志数据
        for i, qa_pair in enumerate(self.state['completed_qa_pairs']):
            question_num = i + 1  # 问题序号从1开始
            log_path = qa_pair.get('log_path')
            
            if log_path:
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        question_log_data[question_num] = json.load(f)
                    print(f"已加载问题 {question_num} 的日志数据")
                except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
                    print(f"警告：无法读取问题 {question_num} 的日志文件 {log_path}: {e}")
                    question_log_data[question_num] = None
        
        def extract_file_paths_from_citation(citation_match):
            """解析单个引用标记并返回对应的文档名列表"""
            citation_text = citation_match.group(0)  # 完整的匹配文本，如 [Q #1, E #16, R #15]
            
            try:
                # 移除方括号
                inner_text = citation_text[1:-1]  # 去掉 [ 和 ]
                
                # 按分号分割不同的问题组，如 "Q #1, R #3; Q #2, R #3 #5"
                question_groups = [group.strip() for group in inner_text.split(';')]
                
                all_file_paths = set()  # 使用set去重
                
                for group in question_groups:
                    # 解析每个问题组，如 "Q #1, E #16, R #15"
                    parts = [part.strip() for part in group.split(',')]
                    
                    current_question = None
                    for part in parts:
                        if part.startswith('Q #'):
                            # 提取问题序号
                            current_question = int(part.replace('Q #', ''))
                        elif current_question and current_question in question_log_data:
                            # 处理实体/关系/文本单元引用
                            log_data = question_log_data[current_question]
                            if log_data:
                                file_paths = self._extract_file_paths_from_part(part, log_data)
                                all_file_paths.update(file_paths)
                
                return list(all_file_paths)
                
            except Exception as e:
                print(f"警告：解析引用标记 {citation_text} 时出错: {e}")
                return []
        
        def replace_citation(match):
            """替换函数：将旧引用格式转换为新的数字格式"""
            nonlocal citation_counter
            
            # 获取当前引用对应的所有文档名
            file_paths = extract_file_paths_from_citation(match)
            
            # 为每个文档名分配序号
            citation_numbers = []
            for file_path in sorted(set(file_paths)):  # 去重并排序
                if file_path not in references_map:
                    references_map[file_path] = citation_counter
                    citation_counter += 1
                citation_numbers.append(references_map[file_path])
            
            # 生成新的引用格式
            if citation_numbers:
                citation_numbers.sort()
                if len(citation_numbers) == 1:
                    return f"[{citation_numbers[0]}]"
                else:
                    return f"[{', '.join(map(str, citation_numbers))}]"
            else:
                return match.group(0)  # 如果无法解析，保持原样
        
        # 使用正则表达式查找和替换所有引用标记
        citation_pattern = r'\[Q #[^\]]+\]'
        processed_report = re.sub(citation_pattern, replace_citation, raw_report)
        
        # 生成参考文献列表
        if references_map:
            # 按序号排序
            sorted_references = sorted(references_map.items(), key=lambda x: x[1])
            
            references_section = "\n\n## 参考文献\n\n"
            for doc_name, ref_num in sorted_references:
                references_section += f"[{ref_num}] {doc_name}\n"
            
            processed_report += references_section
            print(f"已生成 {len(references_map)} 个参考文献")
        
        return processed_report
    
    def _extract_file_paths_from_part(self, part, log_data):
        """从引用部分提取文档路径"""
        file_paths = set()
        
        try:
            if part.startswith('E #'):
                # 处理实体引用，如 "E #16" 或 "E #14-16"
                ids = self._parse_id_range(part.replace('E #', ''))
                entities = log_data.get('entities', [])
                for entity in entities:
                    if str(entity.get('id', '')) in ids:
                        file_path = entity.get('file_path')
                        if file_path:
                            file_paths.update(path.strip() for path in file_path.split(';') if path.strip())
            
            elif part.startswith('R #'):
                # 处理关系引用，如 "R #15" 或 "R #15 #17"  
                ids = self._parse_id_list(part.replace('R #', ''))
                relations = log_data.get('relations', [])
                for relation in relations:
                    if str(relation.get('id', '')) in ids:
                        file_path = relation.get('file_path')
                        if file_path:
                            file_paths.update(path.strip() for path in file_path.split(';') if path.strip())
            
            elif part.startswith('DC #'):
                # 处理文本单元引用
                ids = self._parse_id_list(part.replace('DC #', ''))
                text_units = log_data.get('text_units', [])
                for text_unit in text_units:
                    if str(text_unit.get('id', '')) in ids:
                        file_path = text_unit.get('file_path')
                        if file_path:
                            file_paths.update(path.strip() for path in file_path.split(';') if path.strip())
        
        except Exception as e:
            print(f"警告：解析引用部分 {part} 时出错: {e}")
        
        return file_paths
    
    def _parse_id_range(self, id_str):
        """解析ID范围，如 '14-16' -> ['14', '15', '16']"""
        ids = set()
        if '-' in id_str:
            try:
                start, end = map(int, id_str.split('-'))
                ids.update(str(i) for i in range(start, end + 1))
            except ValueError:
                ids.add(id_str)
        else:
            ids.add(id_str)
        return ids
    
    def _parse_id_list(self, id_str):
        """解析ID列表，如 '15 #17' -> ['15', '17']"""
        # 移除所有 # 符号并按空格分割
        cleaned = id_str.replace('#', '')
        return set(cleaned.split())

    def run(self):
        print(f"开始针对 '{self.state['original_topic']}' 进行研究...")
        self.plan_research()
        self.execute_and_reflect()

        if CONNECTION_TEST_BEFORE_STAGE_CONCLUSION:
            quick_connectivity_test(config().dr_conclusion_model_url, config().dr_conclusion_model_key, config().dr_conclusion_model)

        raw_report = self.synthesize_report()

        # 新增：对原始报告进行润色（严格保留引用标注及其位置）
        if POLISH:
            raw_polish_report = self.polish_report(raw_report)
        else:
            raw_polish_report = raw_report

        # 在生成原始报告后，进行后处理以添加引用
        self.references_map = {}  # 初始化引用映射
        processed_report = self._post_process_report_and_add_references(raw_polish_report, self.references_map)

        return raw_report, processed_report, raw_polish_report


if __name__ == "__main__":
    
    topic = TOPIC
    # 使用了新的参数来初始化 ResearchAgent
    agent = ResearchAgent(
        topic=topic, 
        model_name=config().dr_query_model,
        initial_questions=INITIAL_QUESTIONS_COUNT,
        max_extra_questions=MAX_ADDITIONAL_QUESTIONS
    )
    
    try:
        raw_report, report, raw_polish_report = agent.run()
        print("\n\n========== 最终研究报告 ==========\n")
        print(report)

        # 将报告保存到文件中
        timestamp = time.time()
        timestamp_str = str(timestamp)
        report_filename = f"research_report_{timestamp_str}.txt"
        raw_report_filename = f"raw_research_report_{timestamp_str}.txt"
        raw_polish_report_filename = f"raw_polish_research_report_{timestamp_str}.txt"

        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(f"研究主题：{topic}\n\n")
                f.write(report)
            print(f"\n--- 报告已成功保存至文件: {report_filename} ---")
        except IOError as e:
            print(f"错误：无法将报告写入文件 {report_filename}。错误信息: {e}")

        try:
            with open(raw_report_filename, 'w', encoding='utf-8') as f:
                f.write(f"研究主题：{topic}\n\n")
                f.write(raw_report)
            print(f"\n--- 原始报告已成功保存至文件: {raw_report_filename} ---")
        except IOError as e:
            print(f"错误：无法将报告写入文件 {raw_report_filename}。错误信息: {e}")

        # 新增：保存润色后的原始报告（保留引用标注及其位置）
        if POLISH:
            try:
                with open(raw_polish_report_filename, 'w', encoding='utf-8') as f:
                    f.write(f"研究主题：{topic}\n\n")
                    f.write(raw_polish_report)
                print(f"\n--- 润色后的原始报告已成功保存至文件: {raw_polish_report_filename} ---")
            except IOError as e:
                print(f"错误：无法将润色后的原始报告写入文件 {raw_polish_report_filename}。错误信息: {e}")
    except Exception as e:
        print(f"研究流程执行出错: {e}")

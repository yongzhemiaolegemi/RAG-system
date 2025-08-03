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

MODEL = "qwen3-235b-a22b-instruct-2507"
INITIAL_QUESTIONS_COUNT = 8 # 初始生成的问题数量
MAX_ADDITIONAL_QUESTIONS = 2 # 最多允许额外增加的问题数量
PERFORM_VERIFICATION = False # 是否执行验证报告的生成
SOURCE_DOCUMENTS_DIR = "./africa2024_raw_files/" # 源文档所在的目录
QUERY_PROMPT = "给出参考文献时，请忠实给出原文档名，不要翻译成中文，也不要把relation等误当作文档名。"
# TOPIC = "需要一个标题为：“非洲生物安全态势研判”的报告。字数大约在2000~3000字。分为三个章节：1.非洲生物安全态势 2.非洲生物安全风险点 3.对我国(中国)应对风险及加强非洲国际合作的建议"
# TOPIC = "需要一个标题为：“非洲生物安全态势研判”的报告。几百字即可"
TOPIC = '''
Prompt：非洲生物安全态势研判报告生成指令
请撰写一篇智库研究报告，标题为：
《非洲生物安全态势研判》
本报告面向政策研究机构内部参考，供政府部门领导研判决策使用。全文建议控制在 3000 字左右，内容要求如下：
________________________________________
📌 报告结构与内容指引
1️⃣ 非洲生物安全整体态势
•	概述非洲当前生物安全形势；
•	可包括传染病流行趋势、生物防控基础能力、关键机构作用、国际合作机制等方面；
•	引用 2024年 9 月以来的最新数据或正式报告支撑分析。
2️⃣ 当前面临的主要生物安全风险与深度分析
•	梳理主要风险类型，并提供实际案例或数据支持；
•	强调风险的系统性与持续性，指出短期内难以缓解的问题；
•	分析风险背后的结构性根源，可结合具体国家进行说明。
3️⃣ 中国应对策略与中非合作建议
•	建议应紧扣前述风险与问题，围绕战略定位、机制建设、能力提升、多边协同等方面提出；
•	所有建议应具备一定可行性与针对性；
•	可结合中国相关政策立场提出合作方向建议，避免泛泛而谈。
________________________________________
📌 写作风格与技术规范
•	使用政策研究风格语言，理性、专业、逻辑清晰；
•	引用权威数据与正式文件时请标注来源与时间；
•	缩略词请在首次出现时注明全称；
•	报告应避免空泛，突出问题导向与政策建议的适用性。
________________________________________
✅ 输出目标：
生成一篇逻辑清晰、数据充分、问题导向明确、建议具有可操作性的智库研究报告，可供政府部门领导研判使用。


'''


def post_to_openai_api(messages, model, stream=False, collect_stream=True, tools=None):
    try:
        client = OpenAI(
            api_key=config().django_llm_key,
            base_url=config().django_llm_url
        )
        
        # 构建API调用参数
        api_params = {
            "model": model,
            "messages": messages,
            # "extra_body": {"enable_thinking": True},
            "stream": stream,
        }
        
        # 如果提供了工具，添加到API调用中
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        print(f"调用API: {config().django_llm_url}")
        print(f"模型: {model}")
        print(f"消息数量: {len(messages)}")
        
        completion = client.chat.completions.create(**api_params)
        
    except Exception as api_error:
        print(f"API调用错误详情: {str(api_error)}")
        print(f"API URL: {config().django_llm_url}")
        print(f"API Key: {config().django_llm_key[:20]}...")
        raise api_error
    
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

def send_query_to_RAG_server(query: str, mode: str = "naive", url: str = config().lightrag_service_url) -> Tuple[str, str] :
    """向RAG服务器发送查询"""
    # POST to url
    # {
    #     "message": "你的查询问题",
    #     "mode": "查询模式"
    # }
    data = {
        "message": query,
        "mode": mode,
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

    def _call_llm(self, system_prompt, user_prompt, tools=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = post_to_openai_api(messages, self.model, stream=False, tools=tools)
            
            content = response['choices'][0]['message'].get('content')
            if content:
                return content.strip()
            else:
                print("警告: LLM响应中没有内容。")
                return ""
        except (OpenAIError, KeyError, IndexError) as e:
            print(f"调用LLM时出错: {e}")
            return "" # 或者返回一个表示失败的特殊字符串

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
            "请严格按照以下格式返回：一个包含问题的JSON数组，不要有其他任何文字或解释。"
            "例如: [\"问题1？\", \"问题2？\", ...]"
        )

        response_text = self._call_llm(system_prompt, user_prompt)
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
            print(f"正在研究问题: {current_question}")

            try:
                rag_answer, log_file_path = send_query_to_RAG_server(QUERY_PROMPT + current_question, mode="naive")
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

            response_text = self._call_llm(system_prompt, user_prompt)
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
            "6. 每个小节的主体是1~2段内容丰富、语言连续的段落，请不要大量分点。\n"
            "7. 尽量多出现一些和时间有关的叙述，用以提升报告的时效性。例如xxxx年xx月这种。\n"
            "8. 在句子后标注引用来自哪个问题的回复。如，如果是来自问题1的回复中的[DC #16]，那么就在句子后面标注[Q #1, DC #16]。如果有多个来源，可以写成[Q #1, DC #4-6 #15 #17]这种。如果一个句子涉及到两个及以上的问题的回复，例如同时涉及到了第一个问题的回复和第二个问题的回复，那么可以写成[Q #1, DC #3; Q #2, DC #3 #5]这种。Q的后面必须带有DC，禁止出现[Q #x] [Q #8; DC #3 #7 #10]这种只有问题编号却无实际内容或者只有内容没有问题编号的标注。"
        )

        final_report = self._call_llm(system_prompt, user_prompt)
        print("\n--- 阶段三完成：报告已生成。---")
        return final_report

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

    def verify_report(self, report_text):
        """
        验证报告中的引用是否与原文相符。

        Args:
            report_text (str): 完整的报告文本。

        Returns:
            str: 验证结果报告。
        """
        print("\n--- 阶段四：验证报告内容 ---")

        # 1. 解析报告，提取正文和引用
        # 假设报告结构是：主题、正文、参考文献
        try:
            # 移除主题和参考文献，只保留正文
            body_text = report_text.split("\n\n## 参考文献\n\n")[0]
            if self.state['original_topic'] in body_text:
                body_text = body_text.replace(f"研究主题：{self.state['original_topic']}\n\n", "")

            # 提取所有引用标记及其所在段落
            # 使用正则表达式查找所有带引用的段落
            # 一个段落被定义为被换行符包围的文本块
            paragraphs = [p.strip() for p in body_text.split('\n\n') if p.strip()]
            citations = defaultdict(list)
            citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'

            for para in paragraphs:
                matches = re.findall(citation_pattern, para)
                if matches:
                    # 清理段落，移除引用标记，以便发送给LLM
                    cleaned_para = re.sub(citation_pattern, '', para).strip()
                    for match in matches:
                        # 一个引用标记可能包含多个数字，例如 [1, 2]
                        ref_numbers = [int(n.strip()) for n in match.split(',')]
                        for ref_num in ref_numbers:
                            citations[ref_num].append(cleaned_para)

        except Exception as e:
            return f"解析报告时出错: {e}"

        # 2. 构建验证任务并调用LLM
        verification_results = []
        # 反转 references_map 以便通过引用序号查找文档名
        ref_map = {v: k for k, v in self.references_map.items()}

        for ref_num, paras in citations.items():
            doc_name = ref_map.get(ref_num)
            if not doc_name:
                verification_results.append({
                    "document": f"文档序号 {ref_num}",
                    "status": "无法找到源文档",
                    "details": "",
                    "paragraphs": paras
                })
                continue

            # 从指定目录读取源文档内容
            source_content = f"无法加载源文档 '{doc_name}' 的内容。"
            try:
                # 确保文件名安全，防止目录遍历攻击
                safe_doc_name = os.path.basename(doc_name)
                doc_path = os.path.join(SOURCE_DOCUMENTS_DIR, safe_doc_name + '.txt')
                
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        source_content = f.read()
                else:
                    print(f"警告: 源文件未找到: {doc_path}")
                    verification_results.append({
                        "document": doc_name,
                        "status": "源文件未找到",
                        "path": doc_path,
                        "details": f"尝试定位的源文档未找到: {doc_path}",
                        "paragraphs": paras
                    })
                    continue

            except Exception as e:
                print(f"读取源文件 {doc_name} 时出错: {e}")


            system_prompt = (
                "你是一位严谨的事实核查员。你的任务是判断所提供的段落内容是否与源文档内容一致。"
                "请仔细阅读源文档和段落，然后给出你的判断。"
                "判断结果必须是以下三种之一：True（符合事实）、False（违反事实）、Irrelevant（不相关）。"
                "不要返回任何其他多余的文字或解释。"
            )
            joined_paras = '\n\n'.join(paras)
            user_prompt = (
                f"源文档内容:\n--- --- --- --- ---\n{source_content}\n--- --- --- --- ---\n\n"
                f"待验证段落:\n--- --- --- --- ---\n{joined_paras}\n--- --- --- --- ---"
            )

            retry_count = 0
            while retry_count < 3:
                response = self._call_llm(system_prompt, user_prompt)
                if response in ["True", "False", "Irrelevant"]:
                    status = response
                    details = ""
                    if status in ["False", "Irrelevant"]:
                        # 追问原因
                        reason_prompt_system = "你是一位深入的分析师。请解释为什么前面的段落被判断为违反事实或不相关。请提供具体理由。"
                        reason_prompt_user = (
                            f"源文档内容:\n{source_content}\n\n"
                            f"段落:\n{joined_paras}\n\n"
                            f"判断结果: {status}"
                        )
                        details = self._call_llm(reason_prompt_system, reason_prompt_user)
                    
                    verification_results.append({
                        "document": doc_name,
                        "status": status,
                        "paragraphs": paras,
                        "details": details
                    })
                    break
                else:
                    retry_count += 1
            
            if retry_count == 3:
                verification_results.append({
                    "document": doc_name,
                    "status": "模型无法判断",
                    "paragraphs": paras,
                    "details": f"模型在3次尝试后仍未返回有效结果。最后一次返回值为: {response}"
                })

        # 3. 生成并保存验证报告
        report = "# 研究报告内容验证结果\n\n"
        issues = [res for res in verification_results if res['status'] != 'True']

        if not issues:
            report += "所有内容均已通过验证，未发现问题。\n"
        else:
            report += f"共发现 {len(issues)} 个问题。\n\n"
            for issue in issues:
                if issue['status'] in ["源文件未找到", "无法找到源文档"]:
                    report += f"## 未找到源文档: {issue['document']}\n"
                    if issue.get('path'):
                        report += f"- **尝试定位路径**: {issue['path']}\n"
                    if issue.get('details'):
                        report += f"- **详情**: {issue['details']}\n"
                    report += "- **相关段落**:\n"
                    for para in issue['paragraphs']:
                        report += f"  - {para}\n"
                    report += "\n"
                else:
                    report += f"## 文档: {issue['document']}\n"
                    report += f"- **状态**: {issue['status']}\n"
                    if issue['details']:
                        report += f"- **详情**: {issue['details']}\n"
                    report += "- **相关段落**:\n"
                    for para in issue['paragraphs']:
                        report += f"  - {para}\n"
                    report += "\n"
        
        return report

    def run(self):
        print(f"开始针对 '{self.state['original_topic']}' 进行研究...")
        self.plan_research()
        self.execute_and_reflect()
        raw_report = self.synthesize_report()
        
        # 在生成原始报告后，进行后处理以添加引用
        self.references_map = {} # 初始化引用映射
        processed_report = self._post_process_report_and_add_references(raw_report, self.references_map)
        
        if PERFORM_VERIFICATION:
            # 验证报告内容
            verification_report = self.verify_report(processed_report)
            
            # 将验证报告保存到文件
            try:
                timestamp = time.time()
                verification_filename = f"verification_report_{timestamp}.md"
                with open(verification_filename, 'w', encoding='utf-8') as f:
                    f.write(verification_report)
                print(f"\n--- 验证报告已成功保存至文件: {verification_filename} ---")
            except IOError as e:
                print(f"错误：无法将验证报告写入文件 {verification_filename}。错误信息: {e}")

        return raw_report, processed_report

if __name__ == "__main__":
    
    topic = TOPIC
    # 使用了新的参数来初始化 ResearchAgent
    agent = ResearchAgent(
        topic=topic, 
        model_name=MODEL,
        initial_questions=INITIAL_QUESTIONS_COUNT,
        max_extra_questions=MAX_ADDITIONAL_QUESTIONS
    )
    
    try:
        raw_report, report = agent.run()
        print("\n\n========== 最终研究报告 ==========\n")
        print(report)

        # 将报告保存到文件中
        timestamp = time.time()
        timestamp_str = str(timestamp)
        report_filename = f"research_report_{timestamp_str}.txt"
        raw_report_filename = f"raw_research_report_{timestamp_str}.txt"
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

    except Exception as e:
        print(f"研究流程执行出错: {e}")


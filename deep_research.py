import json
from typing import Tuple
import uuid # 确保 uuid 被导入，用于流式处理中的 tool_call id 生成

from openai import OpenAIError # 用于更精确地捕获 OpenAI 相关异常
import requests
from openai import OpenAI
from utils import config

MODEL = "qwen3-235b-a22b-instruct-2507"
INITIAL_QUESTIONS_COUNT = 5  # 初始生成的问题数量
MAX_ADDITIONAL_QUESTIONS = 2 # 最多允许额外增加的问题数量
QUERY_PROMPT = "给出参考文献时，请忠实给出原文档名，不要翻译成中文，也不要把relation等误当作文档名。"

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

def send_query_to_RAG_server(query: str, mode: str = "hybrid", url: str = config().lightrag_service_url) -> Tuple[str, str] :
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

    def __init__(self, topic, model_name="gpt-4.1-mini-ca", initial_questions=INITIAL_QUESTIONS_COUNT, max_extra_questions=MAX_ADDITIONAL_QUESTIONS):

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
                rag_answer, log_file_path = send_query_to_RAG_server(QUERY_PROMPT + current_question)
                print(f"RAG回答: {rag_answer[:100]}...") # 打印部分回答作为反馈
            except Exception as e:
                print(f"调用RAG服务时出错: {e}")
                rag_answer = "未能从知识库获取相关信息。" # 默认回答

            self.state['completed_qa_pairs'].append({
                "question": current_question,
                "answer": rag_answer
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
            "8. 在句子后标注引用来自哪个文档，例如正文可以这样写：...with AU and Africa CDC leadership and broad stakeholder engagement ([#2], [#3], [#11])，然后在报告的附录这样写：- #1: Africa CDC and WHO collaborations...\n- #2: Africa CDC and EU partnership\n...\nrag系统返回的内容的Reference中包含了完整的文档名，你在附录里面也要包含完整的文档名，不可以偷懒简写。"
        )

        final_report = self._call_llm(system_prompt, user_prompt)
        print("\n--- 阶段三完成：报告已生成。---")
        return final_report

    def run(self):
        print(f"开始针对 '{self.state['original_topic']}' 进行研究...")
        self.plan_research()
        self.execute_and_reflect()
        final_report = self.synthesize_report()
        return final_report

if __name__ == "__main__":
    
    topic = "需要一个标题为：“非洲生物安全态势研判”的报告。分为三个章节：1.非洲生物安全态势 2.非洲生物安全风险点 3.对我国(中国)应对风险及加强非洲国际合作的建议"
    # 使用了新的参数来初始化 ResearchAgent
    agent = ResearchAgent(
        topic=topic, 
        model_name=MODEL,
        initial_questions=INITIAL_QUESTIONS_COUNT,
        max_extra_questions=MAX_ADDITIONAL_QUESTIONS
    )
    
    try:
        report = agent.run()
        print("\n\n========== 最终研究报告 ==========\n")
        print(report)

        # 将报告保存到文件中
        report_filename = f"research_report_{uuid.uuid4().hex[:8]}.txt"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(f"研究主题：{topic}\n\n")
                f.write(report)
            print(f"\n--- 报告已成功保存至文件: {report_filename} ---")
        except IOError as e:
            print(f"错误：无法将报告写入文件 {report_filename}。错误信息: {e}")

    except Exception as e:
        print(f"研究流程执行出错: {e}")


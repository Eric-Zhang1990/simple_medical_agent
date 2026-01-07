import json
import torch
from retriever import AdvanceRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalAgent:
    def __init__(self, retriever):
        self.model_path = r"/demo1\Qwen\Qwen2.5-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype="auto")
        # 如果电脑显存/内存不够时，可以设置torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16)
        self.retriever = retriever
    
    def tool_query_database(self, drug_name):
        # 模拟数据库查询
        database = {
            "阿莫西林": {"price": 20, "stock": 100},
            "布洛芬": {"price": 15, "stock": 50},
            "对乙酰氨基酚": {"price": 10, "stock": 0},
        }
        result = database.get(drug_name, {"price": "未知", "stock": "未知"})
        return f"药品：{drug_name}，价格：{result['price']}元，库存：{result['stock']}件"

    def tool_retrieve_knowledge(self, query):
        context = self.retriever.get_relevant_context(query)
        return context

    def chat(self, user_query):
        system_prompt = """
你是一个医疗专家 Agent。你有两个工具：
1. `search_knowledge`: 用于查询医学知识、症状、禁忌等（输入：问题字符串）。
2. `check_stock`: 用于查询药品的价格和库存（输入：药品名称）。

请分析用户问题，如果需要使用工具，请输出如下 JSON 格式：
{"tool": "工具名称", "args": "参数"}

如果不需要工具，直接回答用户。
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        response_1 = self._generate(messages)
        # 尝试解析 JSON (模拟 Agent 的 Action 步骤)
        try:
            # 简单清洗一下，防止小模型输出多余的字符
            clean_json = response_1.strip().replace("```json", "").replace("```", "")
            action = json.loads(clean_json)
            print(f"action: {action}")
            tool_name = action.get("tool")
            tool_arg = action.get("args")
            
            tool_result = ""
            if tool_name == "search_knowledge":
                print(f"🤖 [Agent Decision] 调用知识库搜索: {tool_arg}")
                tool_result = self.tool_retrieve_knowledge(tool_arg)
            elif tool_name == "check_stock":
                print(f"🤖 [Agent Decision] 调用库存查询: {tool_arg}")
                tool_result = self.tool_query_database(tool_arg)
            
            # 第二次推理：根据工具结果生成最终回答
            final_input = f"工具执行结果：\n{tool_result}\n\n请根据结果回答用户问题：{user_query}"
            print(f"final_input: {final_input}")
            messages.append({"role": "assistant", "content": response_1}) # 保存历史
            messages.append({"role": "user", "content": final_input})
            
            final_response = self._generate(messages)
            return final_response

        except json.JSONDecodeError:
            # 如果模型没输出 JSON，直接返回它的回答（说明它认为不需要工具，或者是闲聊）
            print("没有找到对应工具")
            return response_1
    
    def _generate(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(inputs.input_ids, max_new_tokens=512, temperature=0.1, top_p=0.9)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).split("assistant\n")[-1]


if __name__ == "__main__":
    # 1. 初始化检索器
    retriever = AdvanceRetriever()

    # 2. 初始化 Agent
    agent = MedicalAgent(retriever)

    # 3. 测试场景
    print("--- 场景1: 知识问答 ---")
    print("AI:", agent.chat("阿莫西林有什么副作用？"))
    
    print("\n--- 场景2: 业务查询 ---")
    print("AI:", agent.chat("帮我查查布洛芬还有库存吗？"))
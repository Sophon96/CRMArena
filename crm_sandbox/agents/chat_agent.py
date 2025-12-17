import os
from openai import OpenAI
from typing import Dict, List
import time, traceback
from crm_sandbox.agents.prompts import SCHEMA_STRING, REACT_RULE_STRING, SYSTEM_METADATA, REACT_EXTERNAL_INTERACTIVE_PROMPT, REACT_INTERNAL_INTERACTIVE_PROMPT, REACT_INTERNAL_PROMPT, REACT_EXTERNAL_PROMPT, REACT_PRIVACY_AWARE_EXTERNAL_PROMPT, REACT_PRIVACY_AWARE_EXTERNAL_INTERACTIVE_PROMPT, ACT_PROMPT
from crm_sandbox.agents.utils import parse_wrapped_response

# Default model to use
DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-09-2025"


class ChatAgent:
    def __init__(
        self, schema_obj, model: str = DEFAULT_MODEL, max_turns: int = 20, eval_mode="default", strategy="react", provider="openai", interactive=False, agent_type="internal", privacy_aware_prompt=False
    ):
        # Initialize OpenAI client for Google Gemini via OpenAI API
        # This is done here so that .env is loaded before the client is created
        self.client = OpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        schema = self._build_schema(schema_obj)
        assert strategy in ["react", "act"], "Only react and act strategies supported for now"
        assert agent_type in ["internal", "external"], "Invalid agent type"
        
        if strategy == "react":
            # react strategy
            if not interactive:
                if agent_type == "internal":
                    self.sys_prompt = REACT_INTERNAL_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                else:
                    if privacy_aware_prompt:
                        self.sys_prompt = REACT_PRIVACY_AWARE_EXTERNAL_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                    else:
                        self.sys_prompt = REACT_EXTERNAL_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
            else:
                if agent_type == "internal":
                    self.sys_prompt = REACT_INTERNAL_INTERACTIVE_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                else:
                    if privacy_aware_prompt:
                        self.sys_prompt = REACT_PRIVACY_AWARE_EXTERNAL_INTERACTIVE_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
                    else:
                        self.sys_prompt = REACT_EXTERNAL_INTERACTIVE_PROMPT.format(system_description=schema, system="Salesforce instance") # add strategy template and schema description
        else:
            # act strategy
            self.sys_prompt = ACT_PROMPT.format(system_description=schema, system="Salesforce instance")
        
        self.agent_type = agent_type
        self.original_model_name = model
        self.model = model
        self.eval_mode = eval_mode
        self.max_turns = max_turns
        self.strategy = strategy
        self.info = {}
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.provider = provider
       
        # All models now use Google Gemini 2.5 Flash Lite
        self.model = DEFAULT_MODEL
            
        # assert self.model in ["o1-mini", "o1-preview", "gpt-4o-2024-08-06", "gpt-3.5-turbo-0125"], "Invalid model name"
    
    def _build_schema(self, schema_obj):
        object_description = dict()
        for item in schema_obj:
            object_description[item["object"]] = "\n".join([f"  - {k}: {v}" for k,v in item["fields"].items()])
            
        template = SCHEMA_STRING.format(
            object_names=", ".join(object_description.keys()),
            object_fields="\n".join(
                [f"{obj}\n{fields}" for obj, fields in object_description.items()]
            )
        )
        return template
    
    def reset(self, args):
        if args["metadata"]["required"]:
            self.sys_prompt += SYSTEM_METADATA.format(system_metadata=args["metadata"]["required"], system="Salesforce instance") # add task/query-specific metadata here
        if self.eval_mode == "aided" and "optional" in args["metadata"]:
            self.sys_prompt += "\n" + args["metadata"]["optional"]
        
        # Google Gemini uses standard message format
        self.messages = [{"role": "system", "content": self.sys_prompt.strip()}]
        self.messages.append({"role": "user", "content": args["query"].strip()})
        
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        
    def act(self, env, index=None, temperature=0.0):
        query, metadata = env.reset(task_index=index)
        self.reset({"query": query, "metadata": metadata})
        # print("----")
        # print(self.sys_prompt)
        # print("----")
        total_cost = 0.0
        self.info["observation_sizes"] = []
        done = False
        reward = 0
        
        current_agent_turn = 0
        # for turn_id in range(self.max_turns):
        while current_agent_turn < self.max_turns:
            # Sleep for rate limiting
            time.sleep(1)
            info = {}
            current_agent_turn += 1
            
            # Use OpenAI SDK to call Google Gemini
            res = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.0,
                max_tokens=2000,
                top_p=1.0
            )

            
            message = {
                "role": res.choices[0].message.role,
                "content": res.choices[0].message.content or ""
            }
            
            
            usage = res.usage

            for key in self.usage.keys():
                if key != "cost":
                    if hasattr(usage, key):
                        self.usage[key].append(getattr(usage, key, 0))

            # Google Gemini API via OpenAI doesn't provide cost, set to 0
            self.usage["cost"].append(0)
            
            action = self.message_action_parser(message, self.model)
            print("User Turn:", env.current_user_turn if hasattr(env, 'current_user_turn') else 0, "Agent Turn:", current_agent_turn, "Agent:", message["content"].strip())
            self.messages.append({"role": "assistant", "content": message["content"].strip()})
            if action is None:
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Invalid action",
                    "content":  message["content"].strip()
                }
                info["end_reason"] = self.info["end_reason"]
                if self.strategy == "react":
                    self.messages.append({"role": "user", "content": REACT_RULE_STRING})
                elif self.strategy == "act":
                    from crm_sandbox.agents.prompts import ACT_RULE_STRING
                    self.messages.append({"role": "user", "content": ACT_RULE_STRING})
                continue
            obs, reward, done, info = env.step(action)
            
            if "observation_size" in info:
                self.info["observation_sizes"].append(info["observation_size"])
            if "end_reason" in info: # implies error in query
                self.info["end_reason"] = info["end_reason"]
            # reset counter if previous action is respond
            if action["name"] == "respond":
                current_agent_turn = 0
            if done:
                break
            elif action["name"] == "execute": # execution results from
                self.messages.append({"role": "user", "content": f"Salesforce instance output: {obs}"})
            elif action["name"] == "respond": # respond to simulated user
                self.messages.append({"role": "user", "content": obs})
        
        # Here when either max_turns is reached or submitted
        if not done: 
            if "end_reason" not in info: # no error in last query
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Max turns reached",
                    "content":  message["content"].strip()
                }
        self.info["usage"] = self.usage
        self.info["total_cost"] = sum(cost for cost in self.usage["cost"] if cost is not None)
        self.info["num_turns"] = (env.current_user_turn if hasattr(env, 'current_user_turn') else 0, current_agent_turn + 1)
        return reward

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    @staticmethod
    def message_action_parser(message: Dict[str, str], model_name: str) -> Dict[str, str]:
        action = None
        content = message["content"].strip()
        # if model_name "deepseek-r1":
        #     content = content.split("</think>")[1]
        resp = parse_wrapped_response(r'<execute>(.*?)</execute>', content).strip()
        if resp:
            action = {"name": "execute", "content": resp}
            return action
        
        resp = parse_wrapped_response(r'<respond>(.*?)</respond>', content).strip()
        if resp:
            action = {"name": "respond", "content": resp}
            return action
        return action
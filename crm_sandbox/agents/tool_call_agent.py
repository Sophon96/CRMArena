import json, os, copy
from openai import OpenAI
from typing import Dict, List, Any
import re, traceback, ast, time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from crm_sandbox.agents.prompts import SCHEMA_STRING, SYSTEM_METADATA, NATIVE_FC_PROMPT, CUSTOM_FC_PROMPT, FC_RULE_STRING, FC_FLEX_PROMPT
from crm_sandbox.agents.utils import parse_wrapped_response, fc_prompt_builder

# Default model to use
DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-09-2025"

    
class ToolCallAgent:
    def __init__(
        self, tools, schema_obj, model: str = DEFAULT_MODEL, max_turns: int = 20, eval_mode="default", strategy="tool_call", provider="openai"
    ):
        # Initialize OpenAI client for Google Gemini via OpenAI API
        # This is done here so that .env is loaded before the client is created
        self.client = OpenAI(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        schema = self._build_schema(schema_obj)
        # Clean tools: Remove "returns" field which is not part of OpenAI function calling schema
        # Google Gemini API doesn't accept this field
        self.tools = self._clean_tools(tools)
        
        if strategy == "tool_call":
            # Use standard native function calling prompt
            self.sys_prompt = NATIVE_FC_PROMPT.format(system="Salesforce instance")
        else:
            self.sys_prompt = FC_FLEX_PROMPT.format(system_description=schema, system="Salesforce instance")
            
        self.model = DEFAULT_MODEL  # Always use Google Gemini 2.5 Flash Lite
        self.eval_mode = eval_mode
        self.max_turns = max_turns
        self.usage = {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.provider = provider
            

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
    
    def _clean_tools(self, tools):
        """
        Remove 'returns' field from tool definitions as it's not part of the OpenAI function calling schema.
        Google Gemini API doesn't accept this field even through the OpenAI-compatible endpoint.
        """
        cleaned_tools = []
        for tool in tools:
            tool_copy = copy.deepcopy(tool)
            if isinstance(tool_copy, dict) and "function" in tool_copy:
                # Remove the 'returns' field if present
                if "returns" in tool_copy["function"]:
                    del tool_copy["function"]["returns"]
            cleaned_tools.append(tool_copy)
        return cleaned_tools
    
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
        self.info = {}
        self.info["observation_sizes"] = []
        done = False
        reward = 0
        
        for turn_id in range(self.max_turns):
            time.sleep(1)  # Rate limiting
            info = {}
            # Use OpenAI SDK to call Google Gemini with tools
            res = self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=3500,
                tools=self.tools
            )
            message = {
                "role": res.choices[0].message.role,
                "content": res.choices[0].message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in res.choices[0].message.tool_calls
                ] if res.choices[0].message.tool_calls else None
            }
            usage = res.usage
            
            for key in self.usage.keys():
                if key != "cost":
                    if hasattr(usage, key):
                        self.usage[key].append(getattr(usage, key, 0))
            
            # Google Gemini API via OpenAI doesn't provide cost, set to 0
            self.usage["cost"].append(0)
            
            print("message", message, flush=True)
            action = self.message_action_parser(message)
            print("#", turn_id, "Agent action:", action, flush=True)
            
            
            if action is None:
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Invalid action",
                    "content":  message["content"].strip() if message["content"] else ""
                }
                info["end_reason"] = self.info["end_reason"]
                # if tool_call attempted but failed
                if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and "id" in message["tool_calls"][0]:
                    message["tool_calls"] = message["tool_calls"][:1]
                    self.messages.append(message)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": message["tool_calls"][0]["id"].strip(),
                            "name": message["tool_calls"][0]["function"]["name"].strip(),
                            "content": f"Invalid tool call argument. Please make a valid tool call using the tools provided or submit the final answer using the 'respond' tool."
                        }
                    )
                # if no valid tool_call
                else:
                    self.messages.append(message)
                    self.messages.append({"role": "user", "content": FC_RULE_STRING})
                continue
            else:
                message["tool_calls"] = message["tool_calls"][:1]
                self.messages.append(message)
            
            obs, reward, done, info = env.step(action)
            if "observation_size" in info:
                self.info["observation_sizes"].append(info["observation_size"])
            if "end_reason" in info: # implies error in query
                self.info["end_reason"] = info["end_reason"]
            if done: # implies submit action
                break
            else:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_calls"][0]["id"].strip(),
                        "name": action["name"],
                        "content": obs
                    }
                )
        
        # Here when either max_turns is reached or submitted
        if not done: 
            if "end_reason" not in info: # no error in last query
                self.info["end_reason"] = {
                    "source": "agent",
                    "message": "Max turns reached",
                    "content":  message["content"].strip() if message.get("content") else ""
                }
        self.info["usage"] = self.usage
        self.info["total_cost"] = sum(self.usage["cost"])
        self.info["num_turns"] = turn_id + 1
        return reward

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def message_action_parser(self, message: Dict[str, Any]) -> Dict[str, str]:
        # No llama support, only standard tool calling
        if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0].get("function") is not None:
            tool_call = message["tool_calls"][0]
            try:
                return {
                    "name": tool_call["function"]["name"].strip(),
                    "arguments": json.loads(tool_call["function"]["arguments"].strip()),
                }
            except json.JSONDecodeError:
                pass
        
        return None


"""
AI-powered paper query. Input should be a question. The AI will then:
    1. generate several query (plain text) and keywords.
    2. use keywords search the "quick summary" of the papers in the database for relevant papers.
    3. use the query and keywords to search the rag databses for relevant papers. 
    4. combine the results from 2 and 3, then AI will read the "quick summary" for each paper one by one and excude the unrelated papers.
    5. AI will then read the "quick summary" of the remaining papers, and decide whether to read the searched text in step 3.
    If yes, the AI will read the searched text, "quick summary" and generate detailed information to add to the context.
    6. AI will then generate a final answer to the question based on the context.
"""

SYSTEM_PROMPT = """
You are an AI-powered pageboy agent expert in the field of geology, geochemistry, geophysics and astrophysics with a database of scientific papers as your knowledge base. Your task is to generate answers to user's questions based on the information available in the database. Your answers should be concise, accurate, and relevant to the user's question. You should also provide references to the papers supporting your answers. The references should cite with author-year format during the answer, and list the detailed references list at the end. If the information is not available in the database, you should indicate that you do not have enough information to answer the question.
"""

QUERY_GENERATOR_PROMPT = """
Before answering user's this question, you are asked to check the database first. To do this, you need to generate several queries and keywords, and the description of the information you need to find in the searched papers. queries should be in plain text for rag searching, and keywords should be patterns like "A and B OR C". 
The description should be a brief summary of the information you need to find in the searched papers. You should generate at least 3 queries and 3 keywords, and the description should be a concise overview of the information you are looking for. You should not generate any answers to the user's question at this stage. You should only generate queries, keywords and descriptions for searching the database.

The document of keywords patterns:
## Grammar & Token Rules
- **Tokens:** Space-separated substrings.
- **Exact Match:** Wrapped in double quotes (`"..."`). Treats interior words as a single exact phrase token.
- **Operators:** Case-sensitive `AND`, `OR` (must be uppercase).
- **Implicit Logic:** Spaces between non-operator tokens act as an implicit `AND`.
## Operator Precedence
1. Quotes (`"..."`)
2. `AND` (and implicit spaces)
3. `OR`
*Note: Grouping parentheses `()` are not supported. Logic is evaluated via standard Shunting-yard layout.*
## Examples & AST Interpretations
- `A B C` $\rightarrow$ `A AND B AND C`
- `A AND B OR C` $\rightarrow$ `(A AND B) OR C`
- `A OR B AND C` $\rightarrow$ `A OR (B AND C)`
- `"exact phrase" AND keyword` $\rightarrow$ Evaluates exact substring first, then filters by keyword.

## Requirements for the "Description" Section
The description must define the **precise informational target** to be extracted from a **single** paper. It should NOT look for macro-level literature trends.

1. **Single-Paper Focus (Micro vs. Macro):** Frame the description as an extraction task from *one hypothetical paper*, not a synthesis across multiple papers or years.
2. **Highly Specific Context:** Clearly state the exact parameter, metric, mechanism, or value you need, including its engineering/scientific context.
3. **No Meta-Literature Requests:** Do not ask for "a list of papers," "overviews," or "historical trends."
4. **Diversity & Distinct Focus:** The generated descriptions **must be significantly different from one another**. Do not repeat the same core idea with slight variations in wording. Each description should target a fundamentally different aspect of the problem (e.g., one for the parameters, one for constraints, one for evaluation metrics; or one for models, one for observations, one for implications).

### Good vs. Bad Examples
*   ** BAD (Too broad, targets multiple papers/trends):** 
    * "The methods used to model boundary layers in the last five years."
    * "Papers covering practical aspects like time-stepping algorithms."
*   ** GOOD (Specific, targets concrete data within a single paper):**
    * "The specific empirical parameter or constant used to model the boundary layer thickness."
    * "The explicit mathematical formulation or time-stepping constraint used for stability in the simulation."

Return the queries, keywords and descriptions in the following format:
<queries>
<query>query in plain text for rag searching</query>
<query>another query</query>
<query>another query</query>
</queries>
<keywords>
<keyword>keyword pattern like "A and B OR C"</keyword>
<keyword>another keyword pattern</keyword>
<keyword>yet another keyword pattern</keyword>
</keywords>
<descriptions>
<description>description of the information needed in the searched papers</description>
<description>another description</description>
<description>yet another description</description>
</descriptions>

MAKE SURE TO FOLLOW THE FORMAT EXACTLY!!! USE "less than" not "<" in the content to AVOID POST-PROCESSING ERRORS.DO NOT ADD ANY EXTRA TEXT OR EXPLANATION OUTSIDE OF THE FORMAT. DO NOT REPEAT THE USER'S QUESTION IN YOUR RESPONSE. DO NOT ANSWER THE QUESTION AT THIS STAGE.
"""

INFORMATION_GENERATOR_PROMPT = """
Before answering user's question, you need to check the researched result first. To do this, you need to read the "quick summary" and the searched snippets (may not provided if not been required) of the papers, and generate detailed information to add to the context.
Format of output:
<is_relevant>True if this paper is relevant to any of the information required. False for unrelated to any required information</is_relevant>
<query_informations>
<information>
If relavant to the first description, provide the information in detail by:
<is_relevant>True</is_relevant>
<context>Context of the information relative to the first description</context>
<score>Score how sufficiently the information can be derived/extracted from this paper, on a scale of 0-10 (where 10 means the paper directly and completely supports this information, and 0 means it cannot be derived at all).</score>
else, if not relavant to the first description, keep the <context> and <score> empty, and set <is_relevant> to False
</information>
<information>
xxx
</information>
</query_informations>
<require_snippets>True ONLY IF the "quick summary" indicates the paper contains the target information, but the current quick summary lack the exact technical data, parameters, or proofs needed. Set to False if the summary alone already disproves relevance, or contains sufficient information.</require_snippets>
<require_full>True ONLY IF ALL of the following conditions are strictly met:
1. You have already read the snippets.
2. The snippets confirm the exact information exists, but the high-density core data (e.g., complete mathematical proofs, full tables, or complex multi-step methodologies) is missing.
3. It is absolutely impossible to synthesize the final answer without this full text.
Otherwise, set to False. Use this flag with extreme caution.</require_full>

MAKE SURE TO FOLLOW THE FORMAT EXACTLY!!! DO NOT ADD ANY EXTRA TEXT OR EXPLANATION OUTSIDE OF THE FORMAT. DO NOT REPEAT THE USER'S QUESTION IN YOUR RESPONSE. DO NOT ANSWER THE QUESTION AT THIS STAGE.
"""

ENHANCED_PROMPT_ADDING_SNIPPETS = """
REGENERATE THE OUTPUT with provided snippets again. MAKE SURE TO FOLLOW THE FORMAT EXACTLY!!! DO NOT ADD ANY EXTRA TEXT OR EXPLANATION OUTSIDE OF THE FORMAT. DO NOT REPEAT THE USER'S QUESTION IN YOUR RESPONSE. DO NOT ANSWER THE QUESTION AT THIS STAGE. 
"""

ENHANCED_PROMPT_ADDING_FULLTEXT = """
After reading the provided full text of the paper, please regenerate the output of the required information, and provide additional information which may also helpfull to answer the user's question.

Output format:
<is_relevant>True if this paper is relevant to any of the information required. False for unrelated to any required information</is_relevant>
<query_informations>
<information>
If relavant to the first description, provide the information in detail by:
<is_relevant>True</is_relevant>
<context>Context of the information relative to the first description</context>
<score>Score how sufficiently the information can be derived/extracted from this paper, on a scale of 0-10 (where 10 means the paper directly and completely supports this information, and 0 means it cannot be derived at all).</score>
else, if not relavant to the first description, keep the <context> and <score> empty, and set <is_relevant> to False
</information>
<information>
xxx
</information>
</query_informations>
<additional_informations>
<additional_information>Additional information that may help to answer the user's question, plain text, not add other sub-elements</additional_information>
<additional_information>Another additional information</additional_information>
</additional_informations>

MAKE SURE TO FOLLOW THE FORMAT EXACTLY!!! DO NOT ADD ANY EXTRA TEXT OR EXPLANATION OUTSIDE OF THE FORMAT. DO NOT REPEAT THE USER'S QUESTION IN YOUR RESPONSE. DO NOT ANSWER THE QUESTION AT THIS STAGE.
"""

ANSWER_GENERATOR_PROMPT = """
After a search and reading the relevant informations, please generate a final answer to the user's question based on the context. Your answer should be concise, accurate, and relevant to the user's question. You should cite references with author-year format to support your claims and list the detailed references list at the end. If the information is not available in the database, you should indicate that you do not have enough information to answer the question.
"""


from .config import get_config
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
import xml.etree.ElementTree as ET
from typing import Self, Any, Tuple, List, Dict
from dataclasses import dataclass, field
import os

ds_api_key = get_config("ds_api_key")
ds_url = get_config("ds_url")
flash_name = get_config("model_name_cheap")
pro_name = get_config("model_name_professional")
client = OpenAI(api_key=ds_api_key, base_url=ds_url)

@dataclass
class Information_from_paper:
    is_relevant: bool
    context: str
    score: int

@dataclass
class Informations_from_paper:
    is_relevant: bool = False
    require_snippets: bool = False
    require_full: bool = False
    informations: List[Information_from_paper] = field(default_factory=list)
    additional_informations: List[str] = field(default_factory=list)

class PaperQueryChatter:
    def __init__(self, history_hash: str|None = None, max_paper_num: int = 25, max_information_num: int = 50):
        self.client = client
        self.flash_name = flash_name
        self.pro_name = pro_name
        self.current_database_query: Tuple[List[str], List[str], List[str]]|None = None
        self.informations: List[str] = []
        self.context: List[ChatCompletionMessageParam] = []
        self.pad: Dict[int, List[str]] = {}
        self.pad_keys: List[int] = []
        self.max_paper_num = max_paper_num
        self.max_information_num = max_information_num
        self.analyzed_papers: Dict[int, List[str]] = {}
        self.analyzed_papers_keys: List[int] = []
        self.old_informations: List[str] = []
        if history_hash is not None:
            self.history_hash = history_hash
        else:
            import hashlib
            import time
            self.history_hash = hashlib.md5(str(time.time()).encode()).hexdigest()
        # 检查是否有历史记录，如果有，加载历史记录
        self.history_file = f"history_{self.history_hash}.json"
        #print(f"History file: {self.history_file}")
        if os.path.exists(self.history_file):
            try:
                #print(f"Loading history from {self.history_file}...")
                self.load_history()
            except Exception as e:
                print(f"Failed to load history from {self.history_file}: {e}")
                self.initial_history()
        else:
            self.initial_history()
    
    def initial_history(self):
        system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": SYSTEM_PROMPT}
        self.context = [system_message]
    
    def save_history(self) -> None:
        import json
        history_data = {
            "context": self.context,
            "current_database_query": self.current_database_query,
            "informations": self.informations,
            "pad": self.pad,
            "pad_keys": self.pad_keys,
            "analyzed_papers": self.analyzed_papers,
            "analyzed_papers_keys": self.analyzed_papers_keys,
            "old_informations": self.old_informations
        }
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
    
    def load_history(self) -> None:
        import json
        with open(self.history_file, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        context_from_json = history_data.get("context", [])
        # Convert the loaded context to the appropriate types
        self.context = []
        for message in context_from_json:
            role = message.get("role")
            content = message.get("content")
            if role and content:
                if role == "system":
                    system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": content}
                    self.context.append(system_message)
                elif role == "user":
                    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": content}
                    self.context.append(user_message)
                elif role == "assistant":
                    assistant_message: ChatCompletionMessageParam = {"role": "assistant", "content": content}
                    self.context.append(assistant_message)
        
        current_database_query_from_json = history_data.get("current_database_query", None)
        if current_database_query_from_json is not None:
            queries = current_database_query_from_json[0] if len(current_database_query_from_json) > 0 else []
            keywords = current_database_query_from_json[1] if len(current_database_query_from_json) > 1 else []
            descriptions = current_database_query_from_json[2] if len(current_database_query_from_json) > 2 else []
            self.current_database_query = (queries, keywords, descriptions)
        else:
            self.current_database_query = None

        informations_from_json = history_data.get("informations", [])
        #print(informations_from_json)
        self.informations = []
        for info in informations_from_json:
            self.informations.append(info)

        pad_json = history_data.get("pad", {})
        self.pad = {}
        for key, value in pad_json.items():
            self.pad[int(key)] = value
        pad_keys_json = history_data.get("pad_keys", [])
        self.pad_keys = []
        for key in pad_keys_json:
            self.pad_keys.append(int(key))
        
        analyzed_papers_json = history_data.get("analyzed_papers", {})
        self.analyzed_papers = {}
        for key, value in analyzed_papers_json.items():
            self.analyzed_papers[int(key)] = value
        analyzed_papers_keys_json = history_data.get("analyzed_papers_keys", [])
        self.analyzed_papers_keys = []
        for key in analyzed_papers_keys_json:
            self.analyzed_papers_keys.append(int(key))
        
        old_informations_from_json = history_data.get("old_informations", [])
        self.old_informations = []
        for info in old_informations_from_json:
            self.old_informations.append(info)

    def _parse_queries_keywords_descriptions(self, content: str) -> Tuple[List[str], List[str], List[str]]:
        # 防呆处理，如果content无法当成xml解析，尝试简单修复（例如，删除未被<xx></xx>包裹的字符串，添加总<xml></xml>等），再解析
        def _extract_texts(elements: list[Any]) -> List[str]:
                return [text for text in (element.text for element in elements) if text is not None]
        if not content.startswith("<root>"):
            content = f"<root>{content}</root>"
        try:
            root = ET.fromstring(content)
            queries = _extract_texts(root.findall('queries/query'))
            keywords = _extract_texts(root.findall('keywords/keyword'))
            descriptions = _extract_texts(root.findall('descriptions/description'))
            return queries, keywords, descriptions
        except ET.ParseError:
            print(content)
            # 尝试简单修复
            import re
            content_fixed = re.sub(r'(?<!<)(\w+)(?!>)', '', content)  # 删除未被<xx></xx>包裹的字符串
            content_fixed = f"<root>{content_fixed}</root>"  # 添加总<root></root>
            try:
                root = ET.fromstring(content_fixed)
                queries = _extract_texts(root.findall('queries/query'))
                keywords = _extract_texts(root.findall('keywords/keyword'))
                descriptions = _extract_texts(root.findall('descriptions/description'))
                return queries, keywords, descriptions
            except ET.ParseError:
                raise ValueError("Failed to parse queries, keywords, and descriptions from the content.")
    
    def add_query_context(self, query: str) -> Self:
        user_message: ChatCompletionUserMessageParam = {"role": "user", "content": "Question from the user: " + query}
        self.context.append(user_message)
        return self
    
    def generate_queries_and_keywords(self) -> Self:
        # 需要清空之前的current_database_query，pad 和informations，因为新的query可能会导致之前的结果不再适用
        self.current_database_query = None
        self.pad = {}
        self.pad_keys = []
        self.old_informations = self.informations.copy()
        self.informations = []
        prompt_message: ChatCompletionUserMessageParam = {"role": "user", "content": QUERY_GENERATOR_PROMPT}
        response = self.client.chat.completions.create(
            model=self.flash_name,
            messages=self.context + [prompt_message],
            temperature=0.15,
            max_tokens=4096,
            stream = False,
            extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
        )
        content = response.choices[0].message.content
        assert content is not None, "No content returned from the model."
        queries, keywords, descriptions = self._parse_queries_keywords_descriptions(content)
        self.current_database_query = (queries, keywords, descriptions)
        self.save_history()
        return self
    
    def query_database(self) -> Self:
        if self.current_database_query is None:
            raise ValueError("No queries and keywords generated. Please run generate_queries_and_keywords() first.")
        queries, keywords, _ = self.current_database_query
        query_results = {}
        query_keys = []
        from .rag import rag_database_searching
        from .storage import search_quick_summary_by_key_words
        rag_results_dict, rag_results_keys = rag_database_searching(queries, keywords)
        quick_summary_results = {}
        for keyword in keywords:
            quick_summary_results.update(search_quick_summary_by_key_words(keyword))
        
        for rag_key in rag_results_keys:
            if rag_key in quick_summary_results:
                query_results[rag_key] = [
                    "<quick_summary>" + quick_summary_results[rag_key] + "</quick_summary>"
                    ] + [
                    "<rag_snippet>" + snippet + "</rag_snippet>" for snippet in rag_results_dict[rag_key]
                    ]
            else:
                from .storage import Paper
                paper = Paper.get_by_id(rag_key)
                if paper is not None:
                    query_results[rag_key] = [
                        "<quick_summary>" + paper.quick_summary + "</quick_summary>"
                        ] + [
                        "<rag_snippet>" + snippet + "</rag_snippet>" for snippet in rag_results_dict[rag_key]
                        ]
                else:
                    query_results[rag_key] = [
                        "<rag_snippet>" + snippet + "</rag_snippet>" for snippet in rag_results_dict[rag_key]
                        ]
            query_keys.append(rag_key)
        
        for key in quick_summary_results:
            if key not in query_results:
                query_results[key] = ["<quick_summary>" + quick_summary_results[key] + "</quick_summary>"]
                query_keys.append(key)
        self.pad = query_results
        self.pad_keys = query_keys
        self.save_history()
        return self

    def _analyze_one_paper(self, paper_id: int) -> List[str]|None:
        def _parse_information_xmls(content: str) -> Informations_from_paper:
            print(content)
            import re
            
            # 初始化返回的 DataClass
            result = Informations_from_paper()
            
            # --- 步骤 1: 基础防呆清洗 ---
            # 移除可能包含在 ```xml ... ``` 中的标记
            content = re.sub(r"```xml\s*", "", content)
            content = re.sub(r"```\s*", "", content)
            content = content.strip()
            
            # LLM 返回的可能不是单一根节点，包裹一个 <root> 标签以符合 XML 规范
            if not content.startswith("<root>"):
                xml_data = f"<root>{content}</root>"
            else:
                xml_data = content

            try:
                # 尝试标准 XML 解析
                root = ET.fromstring(xml_data)
                
                # 解析 is_relevant
                is_rel_elem = root.find("is_relevant")
                if is_rel_elem is not None:
                    result.is_relevant = is_rel_elem.text.strip().lower() == "true" if is_rel_elem.text else False
                
                # 解析 require_snippets 与 require_full
                req_snip_elem = root.find("require_snippets")
                if req_snip_elem is not None:
                    result.require_snippets = req_snip_elem.text.strip().lower() == "true" if req_snip_elem.text else False
                    
                req_full_elem = root.find("require_full")
                if req_full_elem is not None:
                    result.require_full = req_full_elem.text.strip().lower() == "true" if req_full_elem.text else False
                
                # 解析 query_informations 中的每一项 information
                query_infos = root.find("query_informations")
                if query_infos is not None:
                    for info_node in query_infos.findall("information"):
                        is_rel_sub = info_node.find("is_relevant")
                        sub_relevant = is_rel_sub.text.strip().lower() == "true" if (is_rel_sub is not None and is_rel_sub.text) else False
                        
                        ctx_elem = info_node.find("context")
                        context_val = ctx_elem.text.strip() if (ctx_elem is not None and ctx_elem.text) else ""
                        
                        score_elem = info_node.find("score")
                        try:
                            score_val = int(score_elem.text.strip()) if (score_elem is not None and score_elem.text) else 0
                        except ValueError:
                            score_val = 0
                            
                        result.informations.append(
                            Information_from_paper(is_relevant=sub_relevant, context=context_val, score=score_val)
                        )
                
                # 解析 additional_informations
                add_infos = root.find("additional_informations")
                if add_infos is not None:
                    for add_node in add_infos.findall("additional_information"):
                        if add_node.text:
                            result.additional_informations.append(add_node.text.strip())
                            
            except ET.ParseError:
                # --- 步骤 2: 强力容错兜底（当 XML 标签闭合残缺或格式混乱时） ---
                # 强行提取全局标志
                is_rel_match = re.search(r"<is_relevant>(.*?)</is_relevant>", content, re.IGNORECASE)
                if is_rel_match:
                    result.is_relevant = is_rel_match.group(1).strip().lower() == "true"
                    
                req_snip_match = re.search(r"<require_snippets>(.*?)</require_snippets>", content, re.IGNORECASE)
                if req_snip_match:
                    result.require_snippets = req_snip_match.group(1).strip().lower() == "true"
                    
                req_full_match = re.search(r"<require_full>(.*?)</require_full>", content, re.IGNORECASE)
                if req_full_match:
                    result.require_full = req_full_match.group(1).strip().lower() == "true"
                
                # 强行用正则切分出所有的 <information> 块
                info_blocks = re.findall(r"<information>(.*?)</information>", content, re.DOTALL | re.IGNORECASE)
                for block in info_blocks:
                    sub_rel_match = re.search(r"<is_relevant>(.*?)</is_relevant>", block, re.IGNORECASE)
                    sub_relevant = sub_rel_match.group(1).strip().lower() == "true" if sub_rel_match else False
                    
                    ctx_match = re.search(r"<context>(.*?)</context>", block, re.DOTALL | re.IGNORECASE)
                    context_val = ctx_match.group(1).strip() if ctx_match else ""
                    
                    score_match = re.search(r"<score>(.*?)</score>", block, re.IGNORECASE)
                    try:
                        score_val = int(score_match.group(1).strip()) if score_match else 0
                    except ValueError:
                        score_val = 0
                        
                    result.informations.append(
                        Information_from_paper(is_relevant=sub_relevant, context=context_val, score=score_val)
                    )
                
                # 强行提取追加信息
                add_blocks = re.findall(r"<additional_information>(.*?)</additional_information>", content, re.DOTALL | re.IGNORECASE)
                for add_block in add_blocks:
                    result.additional_informations.append(add_block.strip())
            
            return result
        
        def generate_information_list(informations_from_paper: Informations_from_paper) -> List[str]:
            information_list = []
            for info in informations_from_paper.informations:
                if info.is_relevant:
                    information_list.append(f"<information>{info.context}</information><sufficiency_score>{info.score}</sufficiency_score><source_paper_id>{paper_id}</source_paper_id>")
            for additional_info in informations_from_paper.additional_informations:
                information_list.append(f"<additional_information>{additional_info}</additional_information><source_paper_id>{paper_id}</source_paper_id>")
            return information_list
        if paper_id not in self.pad:
            raise ValueError(f"Paper ID {paper_id} not found in the pad. Please run query_database() first.")
        contexts = self.pad[paper_id]
        first_context = contexts[0] if len(contexts) > 0 else ""
        if first_context.startswith("<quick_summary>"):
            quick_summary = first_context
        else:
            quick_summary = ""
        rag_snippets = [context for context in contexts if context.startswith("<rag_snippet>")]
        if quick_summary == "" and len(rag_snippets) == 0:
            raise ValueError(f"No quick summary or rag snippets found for paper ID {paper_id}.")
        
        informations_xmls = ["<information>" + info + "</information>" for info in self.informations]
        information_total_xml = "<query_informations>" + "".join(informations_xmls) + "</query_informations>"

        query_message: ChatCompletionMessageParam = {"role": "assistant", "content": information_total_xml}

        skip_check_snippets = False
        if quick_summary == "":
            skip_check_snippets = True
        
        first_prompt_message: ChatCompletionUserMessageParam = {"role": "user", "content": INFORMATION_GENERATOR_PROMPT}
        if skip_check_snippets:
            first_paper_message: ChatCompletionUserMessageParam = {"role": "user", "content": "No quick summary available for this paper, the following are the snippets from the RAG search: " + "".join(rag_snippets)}
        else:
            first_paper_message: ChatCompletionUserMessageParam = {"role": "user", "content": quick_summary}
        
        first_output = self.client.chat.completions.create(
            model=self.flash_name,
            messages=self.context + [query_message, first_prompt_message, first_paper_message],
            temperature=0.15,
            max_tokens=4096*4,
            stream = False,
            extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
        )
        first_content = first_output.choices[0].message.content
        assert first_content is not None, "No content returned from the model."
        informations = _parse_information_xmls(first_content)
        if not informations.is_relevant:
            return None
        
        # 看看是否需要阅读rag snippets
        if not skip_check_snippets and not informations.require_snippets:
            return generate_information_list(informations)
        
        if skip_check_snippets and not informations.require_full:
            return generate_information_list(informations)

        old_messages = self.context + [query_message, first_prompt_message, first_paper_message, {"role": "assistant", "content": first_content}]
        # 如果需要查询rag snippets:
        if not skip_check_snippets and informations.require_snippets:
            snippet_message: ChatCompletionUserMessageParam = {"role": "user", "content": "The following are the snippets from the RAG search: " + "".join(rag_snippets)}
            second_prompt_message: ChatCompletionUserMessageParam = {"role": "user", "content": ENHANCED_PROMPT_ADDING_SNIPPETS}
            second_output = self.client.chat.completions.create(
                model=self.flash_name,
                messages=old_messages + [snippet_message, second_prompt_message],
                temperature=0.15,
                max_tokens=4096*4,
                stream = False,
                extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
            )
            second_content = second_output.choices[0].message.content
            old_messages += [snippet_message, second_prompt_message, {"role": "assistant", "content": second_content}]
            assert second_content is not None, "No content returned from the model."
            informations = _parse_information_xmls(second_content)
            if not informations.is_relevant:
                return None
            if not informations.require_full:
                return generate_information_list(informations)
        
        # 如果需要查询全文
        if informations.require_full:
            from .storage import Paper
            paper = Paper.get_by_id(paper_id)
            if paper is None:
                raise ValueError(f"Paper ID {paper_id} not found in the database.")
            full_text = paper.get_md_text()
            if full_text is None or full_text.strip() == "":
                return generate_information_list(informations)
            full_text_message: ChatCompletionUserMessageParam = {"role": "user", "content": "The following is the full text of the paper: " + full_text}
            third_prompt_message: ChatCompletionUserMessageParam = {"role": "user", "content": ENHANCED_PROMPT_ADDING_FULLTEXT}
            third_output = self.client.chat.completions.create(
                model=self.flash_name,
                messages=old_messages + [full_text_message, third_prompt_message],
                temperature=0.15,
                max_tokens=4096*4,
                stream = False,
                extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
            )
            third_content = third_output.choices[0].message.content
            assert third_content is not None, "No content returned from the model."
            informations = _parse_information_xmls(third_content)
            if not informations.is_relevant:
                return None
            return generate_information_list(informations)

    def analyze_papers(self) -> Self:
        analyzed_paper_count = 0
        if not self.pad_keys:
            raise ValueError("No papers found in the pad. Please run query_database() first.")
        for paper_id in self.pad_keys.copy():
            paper_informations = self._analyze_one_paper(paper_id)
            if paper_informations is not None:
                self.informations.extend(paper_informations)
            #move the analyzed paper to the analyzed_papers list
            self.analyzed_papers[paper_id] = self.pad[paper_id]
            self.analyzed_papers_keys.append(paper_id)
            #remove the analyzed paper from the pad
            del self.pad[paper_id]
            #remove the analyzed paper from the pad_keys
            self.pad_keys.remove(paper_id)
            self.save_history()
            analyzed_paper_count += 1
            if analyzed_paper_count >= self.max_paper_num and len(self.informations) >= self.max_information_num:
                print(f"Enough information have been collected. Stopping analysis after {analyzed_paper_count} papers and {len(self.informations)} informations.")
                break
            if analyzed_paper_count >= self.max_paper_num * 2 or len(self.informations) >= self.max_information_num * 2:
                print(f"Reached the maximum limit of analyzed papers or informations. Stopping analysis after {analyzed_paper_count} papers and {len(self.informations)} informations.")
                break 
        return self
    
    def generate_final_answer(self, generation_by_pro=False) -> Self:
        if not self.informations:
            raise ValueError("No informations found. Please run analyze_papers() first.")
        request_information_xmls = ["<information>" + request + "</information>" for request in self.current_database_query[2]] if self.current_database_query is not None else []
        request_information_xml = "<request_informations>" + "".join(request_information_xmls) + "</request_informations>"
        self.context.append({"role": "assistant", "content": request_information_xml})
        information_total_xml = "<queried_informations>" + "".join(self.informations) + "</queried_informations>"
        bib_infos = []
        for paper_id in self.analyzed_papers_keys:
            from .storage import Paper
            paper = Paper.get_by_id(paper_id)
            bib_info = paper.bib_info() if paper is not None else f"Paper ID {paper_id} not found."
            bib_infos.append(f"<paper_id>{paper_id}</paper_id><bib_info>{bib_info}</bib_info>")
        bib_info_xml = "<bib_infos>" + "".join(bib_infos) + "</bib_infos>"
        query_message: ChatCompletionUserMessageParam = {"role": "user", "content": information_total_xml+bib_info_xml}
        self.context.append(query_message)
        final_prompt_message: ChatCompletionUserMessageParam = {"role": "user", "content": ANSWER_GENERATOR_PROMPT}
        self.context.append(final_prompt_message)
        final_output = self.client.chat.completions.create(
            model=self.flash_name if not generation_by_pro else self.pro_name,
            messages=self.context,
            temperature=0.15,
            max_tokens=4096*4,
            stream = False,
            extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
        )
        final_content = final_output.choices[0].message.content
        assert final_content is not None, "No content returned from the model."
        self.context.append({"role": "assistant", "content": final_content})
        self.save_history()
        return self
    
    def generate_answer_directly(self, use_pro: bool = False) -> Self:
        final_output = self.client.chat.completions.create(
            model=self.flash_name if not use_pro else self.pro_name,
            messages=self.context,
            temperature=0.15,
            max_tokens=4096*4,
            stream = False,
            extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
        )
        final_content = final_output.choices[0].message.content
        assert final_content is not None, "No content returned from the model."
        self.context.append({"role": "assistant", "content": final_content})
        self.save_history()
        return self
    
    def chat(self, query: str, use_pro: bool = False, use_query: bool = False) -> str:
        self.add_query_context(query)
        if use_query:
            self.generate_queries_and_keywords()
            self.query_database()
            self.analyze_papers()
            self.generate_final_answer(generation_by_pro=use_pro)
        else:
            self.generate_answer_directly(use_pro=use_pro)
        return self.context[-1]["content"] # type: ignore

if __name__ == "__main__":
    #pq = PaperQueryChatter(history_hash="83b8f475b43a0110314d72d1664a5169")
    #pq.add_query_context("What are the latest advancements in high temperature Rb isotopes geochemistry?")
    #pq.generate_queries_and_keywords()
    #pq.query_database()
    #pq.analyze_papers()
    #pq.generate_final_answer()
    #print(pq.context[-1]["content"]) # type: ignore
    pq = PaperQueryChatter(history_hash="364325b5e9a6858004c22c27889e5d34")
    #print(pq.informations)
    #pq.generate_final_answer(generation_by_pro=True)
    #print(pq.context[-1]["content"]) # type: ignore
    result=pq.chat("行星形成过程中Rb同位素变重的具体物理机制是什么？能和比较行星学结果符合吗？其他同位素证据能佐证吗？", use_pro=True, use_query=True)
    print(result)
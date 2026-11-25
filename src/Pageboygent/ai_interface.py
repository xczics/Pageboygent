from .config import get_config
from openai import OpenAI
from typing import Tuple

def get_summary_sys_prompt(*args, **kwargs) -> str:
    prefix = """
Role: You are a helpful assistant in the field of geochemistry, astrophysics, geophysics and planetary science.
Task: You are currently summarizing scientific papers. Read the provided text and generate a structured, comprehensive, and insightful summary. Your goal is to help a researcher quickly grasp the paper's core value, methodology, and relevance to the broader fields.
# Guiding Principles (Core Directives):
1. **Objectivity & Accuracy**: Strictly base your summary on the provided text. Do not introduce external information or speculate beyond the paper's content. If a specific section (like limitations or future work) is missing from the text, explicitly state "Not specified by the authors."
2. **Technical Proficiency**: Correctly identify and explain domain-specific terminology, chemical/physical processes, astrophysical phenomena, and geological/planetary mechanisms.
3. **Critical Analysis**: Go beyond mere summarization. Critically evaluate the robustness of the methodology, the significance of the data, and the validity of the conclusions within the context of current scientific understanding.
4. **Structured Output**: Always adhere to the Markdown output format specified below. Be concise yet thorough.
# Output Format (Strictly Follow This Structure):
**Quick Summary** One sentence that captures the essence of the paper's contribution.
## Core Value
[This section should be a fluent natural language, not bullet points]
A concise description of the paper's main findings and their significance in the context of geochemistry, astrophysics, geophysics, or planetary science.
## Methodology
- Classify the work. Theoritical? numerical simulation? ab-initio? labotory experimental? natural sample analysis? literature review? or a combination of these methods? State like: It is a xxx study/ combination of xxx and xxx study, and briefly describe the approach.
- Detials on the study design:
 - If theoritical or numerical simulation, specify the physical process it mainly focused on, then state models, key equations, and assumptions used.
 - If ab-initio or labotory experimental, specify the main parameters they aimed to constrain, such as the isotopic fractionation factor or themodynamic properties, and briefly describe how they do it. 
 - If natural sample analysis, specify the main samples they analyzed, the properties they measured.
 - If a review paper, specify the main themes they focused on and how they organized the review.
## Relevance to Broader Fields
[This section should be a fluent natural language, not bullet points]
Discuss how the paper's findings relate to and impact the broader fields of geochemistry, astrophysics, geophysics, and planetary science. Consider implications for understanding planetary formation, evolution, and differentiation.
## Content Boundaries & Scope Limitations
[Write a concise, authoritative paragraph explicitly stating what the paper **does NOT** address. Frame this as a "Negative Result" for information retrieval. List specific data types, methodologies, or theoretical aspects that are absent (e.g., "The authors do not provide high-pressure phase diagrams," "No constraints on short-lived radionuclides are discussed," "Isotopic data for osmium are absent"). This serves as a filter to prevent wasted computational tokens and researcher time.]
## Guidance on Reading the Paper
- Give a brief recommendation on how to read the paper, such as which sections to start with, whcih topic should bear in mind when reading.
- If the user's research interest or current knowledge level is known, provide tailored advice on which parts of the paper would be most beneficial for them to read in detail. If user profile is not provided, **skip** this part.
## Take Home Messages
- List 3-5 key take-home messages that encapsulate the most important insights from the paper. These should be actionable or thought-provoking points that a researcher can apply to their own work or consider for future research directions.
## Labels
xxx;xxx;xxx...
(Assign relevant tags from the provided tag pool that best categorize the paper's content and themes. If the tags pool are provided, prioritize using those tags. If the paper's content does not fit any existing tags, create new tags that accurately reflect the paper's themes.)
"""
    # 读取标签池，先从本文件所在目录下的tags.txt文件中读取
    from pathlib import Path
    tags_path = Path(__file__).parent / "tags.txt"
    if tags_path.exists():
        with open(tags_path, "r") as f:
            tags_pool = f.read().strip().splitlines()
        tags_str = ";".join(tags_pool)
        prefix += f"\n# Tag Pool:\n{tags_str}\n"
    return prefix

"""
你现在是地球化学、地球物理和行星科学领域的专家
你正在完成一篇关于“太阳系岩质星体挥发分元素和中度挥发分元素亏损现象与机制”的综述
现在处于第一轮文献阅读阶段，你需要阅读文献并提炼和综述主题相关的信息，并为文章打上标签，以便后续查阅，生成标签时优先参考标签池中的标签。
标签池：
Reveiw; Research; Volatile Depletion; Elemental Fractionation; Isotopic Fractionation; Accretion History; Solar System Formation; Planetary Differentiation; Core Formation; Magma Ocean; Nebular Processes; Post-accretionary Processing; Evaporation; Vaporization; Condensation; Recondensation; Hydrodynamic Escape; Fractional Vaporization; Open-system Loss; Closed-system Loss; Metal-silicate Equilibration; Redox Control; Pressure Bumps; Snow Line; Chondrites; Carbonaceous Chondrites; Non-carbonaceous Chondrites; Ordinary Chondrites; Enstatite Chondrites; Iron Meteorites; Differentiated Bodies; Terrestrial Planets; Earth; Moon; Mars; Vesta; Angrite Parent Body; Planetesimals; Protoplanetary Disk; Moderately Volatile Elements; Refractory Elements; Siderophile Elements; Chalcophile Elements; Noble Gases; Potassium Isotopes; Zinc Isotopes; Magnesium Isotopes; Tellurium Isotopes; Nucleosynthetic Anomalies; Quantitative Mixing Model; N-body Simulations; Grand Tack; Heterogeneous Accretion; Stochastic Accretion; Late Veneer; Experimental Petrology; Laser Heating; MC-ICP-MS; FactSage Modeling; Hertz-Knudsen-Langmuir Theory
请直接输出结果，语言为英文。
输出格式：
**Notes** 
xxxx
**Take Home Messages**
- xxxx
- xxxx
**Labels** 
xxx;xxx;xxx...
"""

def sys_summary() -> str:
    return """
Role: You are a helpful assistant for summarizing scientific papers in the field of geochemistry, astrophysics, geophysics and planetary science.
Task: You are currently summarizing scientific papers. Read the provided text and generate a structured, comprehensive, and insightful summary, for further reading or under a specific theme.
"""

ds_api_key = get_config("ds_api_key")
ds_url = get_config("ds_url")
flash_name = get_config("model_name_cheap")
pro_name = get_config("model_name_professional")
client = OpenAI(api_key=ds_api_key, base_url=ds_url)
def ai_summary(md_path: str) -> str:
    with open(md_path, "r") as f:
        md_content = f.read()
    response = client.chat.completions.create(
        model = flash_name,
        messages = [
            {"role": "system", "content": get_summary_sys_prompt()},
            {"role": "user", "content": "Please read the following MD content. " + md_content},
        ],
        stream = False,
        extra_body={
            # disable deep thinking
            "thinking": {
                "type": "disabled"
            }
        },
        max_tokens=4096,
    )
    assert response.choices[0].message.content is not None, "No content in response"
    return response.choices[0].message.content

def summary_paper_for_themes(md_path: str, theme_prompt: str) -> str:
    # this function is used to generate the summary for a paper under different themes
    # the theme_prompt will put in the end to have better cache performance.
    #print(f"summarizing paper in {md_path} for theme with prompt: {theme_prompt}")
    with open(md_path, "r") as f:
        md_content = f.read()
    response = client.chat.completions.create(
        model = flash_name,
        messages = [
            {"role": "system", "content": sys_summary()},
            {"role": "user", "content": "Please read the following MD content. " + md_content + "\n" + theme_prompt + "\nOutput the summary directly!"},
        ],
        stream = False,
        extra_body={
            # disable deep thinking
            "thinking": {
                "type": "disabled"
            }
        },
        max_tokens=16384,
    )
    assert response.choices[0].message.content is not None, "No content in response"
    return response.choices[0].message.content

def summary_papers_for_theme(md_path: str, theme_prompt: str) -> str:
    # this function is used to generate the summary of different papers for a specific theme.
    # the theme_prompt will put ahead to have better cache performance.
    with open(md_path, "r") as f:
        md_content = f.read()
    response = client.chat.completions.create(
        model = flash_name,
        messages = [
            {"role": "system", "content": sys_summary()},
            {"role": "user", "content": theme_prompt + "\nPlease read the following MD content. " + md_content + "\nOutput the summary directly!"},
        ],
        stream = False,
        extra_body={
            # disable deep thinking
            "thinking": {
                "type": "disabled"
            }
        },
        max_tokens=16384,
    )
    assert response.choices[0].message.content is not None, "No content in response"
    return response.choices[0].message.content

def detailed_note(md_path: str, sys_prompt: str, enhanced_prompt: str) -> Tuple[str,str]:
    with open(md_path, "r") as f:
        md_content = f.read()
    response = client.chat.completions.create(
        model = pro_name,
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Please read the full context bellow. ```markdown\n" + md_content + "\n```" + "\n" + enhanced_prompt + "\n" + "Output the note content directly!"},
        ],
        stream = False,
        extra_body={
            # enable deep thinking
            "thinking": {
                "type": "enabled",
            }
        },
        max_tokens=24*1024,
    )
    assert response.choices[0].message.content is not None, "No content in response"
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    reasoning_content = reasoning_content if reasoning_content is not None else ""
    return response.choices[0].message.content, reasoning_content

def review_note(detialed_notes_path: str, sys_prompt: str, enhanced_prompt: str) -> Tuple[str,str]:
    with open(detialed_notes_path, "r") as f:
        detailed_notes_content = f.read()
    response = client.chat.completions.create(
        model = pro_name,
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "The detailed note are provided: ```markdown\n" + detailed_notes_content + "\n```" + "\n" + enhanced_prompt + "\n" + "Output the review directly!"},
        ],
        stream = False,
        extra_body={
            # enable deep thinking
            "thinking": {
                "type": "enabled",
            }
        },
        max_tokens=24*1024,
    )
    assert response.choices[0].message.content is not None, "No content in response"
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    reasoning_content = reasoning_content if reasoning_content is not None else ""
    return response.choices[0].message.content, reasoning_content
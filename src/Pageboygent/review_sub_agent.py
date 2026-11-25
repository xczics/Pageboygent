import xml.etree.ElementTree
from typing import List, Dict,Any
import re
from .storage import Paper
from .ai_interface import detailed_note, review_note
from .config import get_config
from pathlib import Path
import time

detailed_tag_prefix = get_config("tag_for_detailed_reading")
ai_tag = get_config("tag_for_ai_notes")
workstation = Path(get_config("workstation_dir")) / "review_notes"


def preprocess_xml(xml_text: str) -> str:
    return re.sub(
        r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)',
        '&amp;',
        xml_text,
    )

def subproject_pharse(sub_project: str) -> List[Dict[str, Any]]:
    """
    pharse the subprojects from the xml file, and return a list of dict, each dict is one subproject.
    """
    sub_projects = []
    with open(sub_project, "r") as f:
        xml_content = f.read()
    xml_content = preprocess_xml(xml_content)
    root = xml.etree.ElementTree.fromstring(xml_content)
    all_subprojects = root.findall("subproject")
    print(f"Found {len(all_subprojects)} subprojects.")
    out= []
    for sp in all_subprojects:
        # 将子元素转换为字典
        sp_dict: Dict[str, Any] = {}
        for child in sp:
            if child.tag == "biblist":
                sp_dict["biblist"] = []
                for bib in child:
                    bib_dict = {}
                    for bib_child in bib:
                        bib_dict[bib_child.tag] = bib_child.text
                    sp_dict["biblist"].append(bib_dict)
            else:
                child_text = child.text.strip() if child.text else ""
                sp_dict[child.tag] = child_text
        out.append(sp_dict)
    return out

def paste_detailed_notes(Label: str,agent_name: str="PageBoy") -> str:
    from .zt_tools import get_notes_by_tags
    tag_subsection = f"{detailed_tag_prefix}/{Label}"
    # 为避免命名冲突，生成一个8位16进制时间戳
    timestamp = hex(int(time.time()))[2:]
    detailed_read_note = get_notes_by_tags([tag_subsection, ai_tag])
    write_path = workstation / f"detailed_note_{agent_name}_{Label.replace(' ', '_')}_{timestamp}.md"
    with open(write_path, "w") as f:
        f.write(detailed_read_note)
    return str(write_path)

def get_detailed_notes(Label: str,agent_name: str="PageBoy") -> str:
    # 尝试从工作站目录中找到对应的文件
    tag_subsection = f"{detailed_tag_prefix}/{Label}"
    related_files = list(workstation.glob(f"detailed_note_{agent_name}_{Label.replace(' ', '_')}_*.md"))
    if related_files:
        # 如果找到了相关文件，返回最新的一个
        latest_file = max(related_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    else:
        # 如果没有找到相关文件，调用函数生成新的详细笔记
        return paste_detailed_notes(Label, agent_name)

def write_review_note(Label: str, review_context: str, agent_name: str="PageBoy", reasoning_context: str="") -> str:
    tag_subsection = f"{detailed_tag_prefix}/{Label}"
    # 为避免命名冲突，生成一个8位16进制时间戳
    timestamp = hex(int(time.time()))[2:]
    note_path = workstation / f"review_note_{agent_name}_{Label.replace(' ', '_')}_{timestamp}.md"
    with open(note_path, "w") as f:
        f.write(review_context)
    if reasoning_context:
        reasoning_path = workstation / f"reasoning_review_{agent_name}_{Label.replace(' ', '_')}_{timestamp}.md"
        with open(reasoning_path, "w") as f:
            f.write(reasoning_context)
    return str(note_path)

def check_review_note_exists(Label: str, agent_name: str="PageBoy") -> bool:
    tag_subsection = f"{detailed_tag_prefix}/{Label}"
    related_files = list(workstation.glob(f"review_note_{agent_name}_{Label.replace(' ', '_')}_*.md"))
    return len(related_files) > 0

def agent_sub_project(subproject: Dict[str, Any], agent_name: str="PageBoy"):
    if check_review_note_exists(subproject.get("label", ""), agent_name):
        print(f"Review note already exists for subproject: {subproject.get('label', '')}, skipping...")
        return
    from .zt_tools import zot
    biblist = subproject.get("biblist", [])
    assert len(biblist) > 0, "Biblist is empty in subproject: " + subproject.get("label", "")
    Summary_Notes = ""
    for bib in biblist:
        bib_notes = Paper.get_summary_from_note_by_doi(bib.get("doi", ""))
        if bib_notes:
            Summary_Notes += bib_notes + "\n"
    #print("Summary Notes:\n" + Summary_Notes)
    related_collections = []
    for bib in biblist:
        doi = bib.get("doi","")
        if Paper.did_doi_have_note_with_tags(doi, [ai_tag, f"{detailed_tag_prefix}/{subproject.get('label', '')}"]):
            print(f"Detailed note already exists for DOI: {doi}, skipping...")
            continue
        md_path = Paper.get_md_path_by_doi(doi)
        assert md_path is not None, "MD path not found for DOI: " + doi
        subproject_label = subproject.get("label", "")
        sys_read_prompt = subproject.get("readprompt", "")
        enhanced_prompt = subproject.get("enhancedprompt", "")
        sys_prompt = sys_read_prompt + "\n" + \
                "The summary note for the paper list are provided for reference:\n" + \
                "```markdown\n" + Summary_Notes + "\n```\n" + \
                "The note format you are expected to use is \n ```markdown \n " + \
                subproject.get("noteformat", "markdown") + "\n```\n" 
        note_contet, reasoning_content = detailed_note(md_path, sys_prompt, enhanced_prompt)
        Paper.add_note_to_paper_by_doi(doi, note_contet, tags=[
            ai_tag,
            f"{detailed_tag_prefix}/{subproject_label}",
        ])
    detailed_notes = get_detailed_notes(subproject.get("label", ""), agent_name)
    review_sys_prompt = subproject.get("reviewprompt", "")
    review_enhanced_prompt = subproject.get("enhancereviewprompt", None)
    assert review_enhanced_prompt is not None, "Review enhanced prompt is missing in subproject: " + subproject.get("label", "")
    review_context, reasoning_review_content = review_note(detailed_notes, review_sys_prompt, review_enhanced_prompt)
    write_review_note(subproject.get("label", ""), review_context, agent_name, reasoning_review_content)
    print(f"Finished processing subproject: {subproject.get('label', '')}")

def ref():
    sub_projects = subproject_pharse("prompt_sub_projects.xml")
    #print(sub_projects[1].get("readprompt", []))
    #example_subproject = sub_projects[0]
    #example_biblist = example_subproject.get("biblist", [])
    #example_doi = example_biblist[0].get("doi", "")
    #print(example_doi)
    #example_md_path = Paper.get_md_path_by_doi(example_doi)
    #print(example_md_path)
    test_sub_project = sub_projects[0]
    test_biblist = test_sub_project.get("biblist", [])
    Summary_Notes = ""
    for bib in test_biblist:
        bib_notes = Paper.get_summary_from_note_by_doi(bib.get("doi", ""))
        if bib_notes:
            Summary_Notes += bib_notes + "\n"
    print("Summary Notes:\n" + Summary_Notes)
    for test_bib in test_biblist[2:]:
        test_doi = test_bib.get("doi", "")
        test_md_path = Paper.get_md_path_by_doi(test_doi)
        assert test_md_path is not None, "MD path not found for DOI: " + test_doi
        test_label = test_sub_project.get("label", "")
        
        sys_read_prompt = test_sub_project.get("readprompt", "")
        read_format = test_sub_project.get("noteformat", "markdown")
        enhanced_prompt = test_sub_project.get("enhancedprompt", "")
        sys_prompt = sys_read_prompt + "\n" + \
                "The summary note for the paper list are provided for reference:\n" + \
                "```markdown\n" + Summary_Notes + "\n```\n" + \
                "The note format you are expected to use is \n ```markdown \n " + read_format + "\n```\n" + enhanced_prompt
        note_contet, reasoning_content = detailed_note(test_md_path, sys_read_prompt, enhanced_prompt)
    #print(note_contet)
    #print(reasoning_content)
        Paper.add_note_to_paper_by_doi(test_doi, note_contet, tags=[
        "AI Notes",
        f"Detailed Reading/{test_label}",
    ])
        # save reasoning_content and reasoning_content to current dir, generate name with doi
        doi_in_filename = test_doi.replace("/", "_")
        with open(f"reasoning_content_{doi_in_filename}.md", "w") as f:
            f.write(reasoning_content)
        with open(f"note_content_{doi_in_filename}.md", "w") as f:
            f.write(note_contet)

    #print(sys_prompt)

if __name__ == "__main__":
    sub_projects = subproject_pharse("sub_projects2.xml")
    for sub_project in sub_projects[0:1]:
        agent_sub_project(sub_project, "PageBoy")

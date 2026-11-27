# 读取bib文件，并遵循以下流程：
# 1. 解析bib文件，转化为年份-标题-journal-doi-作者的列表
# 2. 筛选出预印本（通过期刊中是否包含arXiv）
# 3. 对于每一个预印本，查找库中是否已经有对应的正式出版物（匹配作者和标题，不严格匹配），如找到，则删除预印本，保留正式出版物；如未找到，保留预印本
# 4. 对于预印本，使用DOI查询Crossref，获取正式出版物的DOI和期刊信息，并更新库中的记录，如果找到不到正式出版物，则保留预印本记录
import re
import bibtexparser
import requests
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding
from typing import List, Dict
def parse_bib_file(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as bib_file:
        bib_database = bibtexparser.load(bib_file, parser=BibTexParser(customization=homogenize_latex_encoding))
    entries = []
    for entry in bib_database.entries:
        year = entry.get('year', '')
        title = entry.get('title', '')
        journal = entry.get('journal', '')
        doi = entry.get('doi', '')
        authors = entry.get('author', '')
        entries.append({
            'year': year,
            'title': title,
            'journal': journal,
            'doi': doi,
            'authors': authors
        })
    return entries
def is_preprint(entry: Dict[str, str]) -> bool:
    return 'arxiv' in entry['journal'].lower()

def find_formal_publication(preprint_entry: Dict[str, str], entries: List[Dict[str, str]]) -> Dict[str, str] | None:
    preprint_title = re.sub(r'\s+', ' ', preprint_entry['title'].lower())
    # 仅匹配第一作者，姓或名任一满足即可
    preprint_authors = [author.strip().lower() for author in preprint_entry['authors'].split(' and ')]
    first_author = preprint_authors[0] if preprint_authors else ''
    first_author_last_name = first_author.split()[-1] if first_author else ''
    first_author_first_name = first_author.split()[0] if first_author else ''
    for entry in entries:
        if entry == preprint_entry:
            continue
        formal_title = re.sub(r'\s+', ' ', entry['title'].lower())
        if preprint_title in formal_title or formal_title in preprint_title:
            formal_authors = [author.strip().lower() for author in entry['authors'].split(' and ')]
            formal_first_author = formal_authors[0] if formal_authors else ''
            if (first_author_last_name in formal_first_author or first_author_first_name in formal_first_author):
                return entry
                # 匹配前打印相关信息，并交互式确认
                #print(f"Preprint: {preprint_entry['title']} by {preprint_entry['authors']}")
                #print(f"Formal Publication: {entry['title']} by {entry['authors']}")
                #confirm = input("Is this the correct formal publication for the preprint? (y/n): ")
                #if confirm.lower() == 'y':
                #     return entry
    return None

def query_crossref(doi: str) -> Dict[str, str] | None:
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'message' in data:
            return data['message']
    return None

def pharse_assertion(assertion: List[Dict[str, str]]) -> Dict[str, str]:
    result = {}
    for item in assertion:
        if 'name' in item and 'value' in item:
            result[item['name']] = item['value']
    return result

def process_bib_file(file_path: str):
    entries = parse_bib_file(file_path)
    preprints = [entry for entry in entries if is_preprint(entry)]
    for preprint in preprints:
        #print(preprint)
        formal_publication = find_formal_publication(preprint, entries)
        if formal_publication:
            print(f"Found formal publication for preprint: {preprint['title']}")
            print(f"Formal publication: {formal_publication['title']} by {formal_publication['authors']}")
            entries.remove(preprint)  # 删除预印本
        else:
            # 如果doi里没有arxiv, 则直接删除期刊字段中的arXiv子串
            if 'arxiv' not in preprint['doi'].lower():
                preprint['journal'] = preprint['journal'].replace('arXiv', '').strip()
                preprint['journal'] = preprint['journal'].replace(':', '').strip()
                preprint['journal'] = preprint['journal'].replace('(', '').strip()
                preprint['journal'] = preprint['journal'].replace(')', '').strip()

            print(f"No formal publication found for preprint: {preprint['title']}")
            if preprint['doi']:
                crossref_data = query_crossref(preprint['doi'])
                if crossref_data:
                    print(f"Crossref data found for DOI {preprint['doi']}")
                    assertion = crossref_data.get('assertion', [])
                    if assertion:
                        info = pharse_assertion(assertion=assertion) # type: ignore
                        doi = info.get('DOI', preprint['doi'])
                        journal = info.get('journal_title', preprint['journal'])
                        title = info.get('article_title', preprint['title'])
                        preprint['doi'] = doi
                        preprint['journal'] = journal
                        preprint['title'] = title
                        print(f"Updated preprint with Crossref data: {preprint['title']} in {preprint['journal']} with DOI {preprint['doi']}")
                    else:
                        print(f"assertion field not found in Crossref data for DOI {preprint['doi']}")
                else:
                    print(f"No Crossref data found for DOI {preprint['doi']}")
            else:
                print(f"No DOI available for preprint: {preprint['title']}")


def export_bib_file(entries: List[Dict[str, str]], output_file: str):
    for entry in entries:
        # 将作者列表转换为字符串
        if isinstance(entry['authors'], list):
            entry['authors'] = ' and '.join(entry['authors'])
        # 默认ENTRYTYPE为article，预印本指定为preprint
        if 'arxiv' in entry['journal'].lower():
            entry['ENTRYTYPE'] = 'preprint'
        else:
            entry['ENTRYTYPE'] = 'article'
        # 设置唯一的ID
        entry['ID'] = re.sub(r'\s+', '_', entry['title'].lower())
        
    bib_database = bibtexparser.bibdatabase.BibDatabase()
    bib_database.entries = entries
    with open(output_file, 'w', encoding='utf-8') as bib_file:
        bibtexparser.dump(bib_database, bib_file)

def clean_and_export_to_bib_file():
    import sys
    # usage: python bibimporter.py path/to/your/bibfile.bib path/to/your/outputfile.bib
    if len(sys.argv) != 3:
        print("Usage: python bibimporter.py path/to/your/bibfile.bib path/to/your/outputfile.bib")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_bib_file(input_file)
    export_bib_file([entry for entry in parse_bib_file(input_file)], output_file)

def clean_and_export_to_zotero():
    from Pageboygent.zt_tools import insert_reference_to_collection
    import sys
    # usage: python bibimporter.py path/to/your/bibfile.bib collection
    if len(sys.argv) != 3:
        print("Usage: python bibimporter.py path/to/your/bibfile.bib collection")
        sys.exit(1)
    input_file = sys.argv[1]
    collection_name = sys.argv[2]
    process_bib_file(input_file)
    entries = parse_bib_file(input_file)
    for entry in entries:
        print(f"Processing entry: {entry['title']} by {entry['authors']}")
        doi = entry['doi']
        journal = entry['journal']
        author = entry['authors']
        title = entry['title']
        type = 'journalArticle'
        insert_reference_to_collection(collection_name=collection_name, type=type, journal=journal, author=author, doi=doi, title=title)

if __name__ == "__main__":
    #clean_and_export_to_bib_file()
    clean_and_export_to_zotero()

            

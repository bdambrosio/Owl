import requests, json
from lxml import etree
url = "http://192.168.1.160:8070/api/processFulltextDocument"
pdf_filepath  = "./arxiv/papers/A_3-miRNA_signature_predicts_prognosis_of_pediatric_and_adolescent_cytogenetically_normal_acute_myeloid_leukemia"
#pdf_filepath = "./arxiv/papers/2311.17970.pdf"

headers = {"Response-Type": "application/xml"}#, "Content-Type": "multipart/form-data"}
# Define the namespace map to handle TEI namespaces
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

def process_xml_table(table):
    start_row_index = -1
    # Try to extract headers using <th> elements; if not present, use the first row's <td> elements
    th_elements = table.xpath('.//tei:tr[1]/tei:th', namespaces=ns)
    if th_elements and len(th_elements)>0 :
        headers = [th.text for th in th_elements]
        start_row_index = 1  # Start from the first row for data rows if headers were found in <th>
    else:
        # If no <th> elements, use the first row's <cell> for headers
        headers = [cell.text for cell in table.xpath('.//tei:row[1]/tei:cell', namespaces=ns)]
        start_row_index = 2  # Start from the second row for data rows, since headers are in the first
        
    #print(f'start_row {start_row_index} headers {headers}')
    # Initialize an empty list to hold each row's data as a dictionary
    table_data = []
        
    # Iterate over each row in the table, excluding the header row
    for row in table.xpath(f'.//tei:row[position()>={start_row_index}]', namespaces=ns):
        # Extract text from each cell (<td>) in the row
        row_data = [cell.text for cell in row.xpath('.//tei:cell', namespaces=ns)]
        if len(row_data) != len(headers): # don't try to process split rows
            continue
        # Create a dictionary mapping each header to its corresponding cell data
        row_dict = dict(zip(headers, row_data))
        table_data.append(row_dict)
            
    # Convert the table data to JSON
    if len(table_data) >0 : # don't store tables with no rows (presumably result of no rows matching headers len)
        #json_data = json.dumps(table_data, indent=4)
        return table_data
    else:
        return None

def parse_pdf(pdf_filepath):
    files= {'input': open(pdf_filepath, 'rb')}
    extract = {"title":'', "authors":'', "abstract":'', "sections":[], "tables": []}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        with open('test.tei.xml', 'w') as t:
            t.write(response.text)
    else:
        print(f'grobid error {response.status_code}, {response.text}')
        return None
    # Parse the XML
    xml_content = response.text
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.fromstring(xml_content.encode('utf-8'))
    # Extract title
    title = tree.xpath('.//tei:titleStmt/tei:title[@type="main"]', namespaces=ns)
    title_text = title[0].text if title else 'Title not found'
    extract["title"]=title_text

    # Extract authors
    authors = tree.xpath('.//tei:teiHeader//tei:fileDesc//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:persName', namespaces=ns)
    authors_list = [' '.join([name.text for name in author.xpath('.//tei:forename | .//tei:surname', namespaces=ns)]) for author in authors]
    extract["authors"]=', '.join(authors_list)

    # Extract abstract
    abstract = tree.xpath('.//tei:profileDesc/tei:abstract//text()', namespaces=ns)
    abstract_text = ''.join(abstract).strip()
    extract["abstract"]=abstract_text

    # Extract major section titles
    # Note: Adjust the XPath based on the actual TEI structure of your document
    section_titles = tree.xpath('./tei:text/tei:body/tei:div/tei:head', namespaces=ns)
    titles_list = [title.text for title in section_titles]
    body_divs = tree.xpath('./tei:text/tei:body/tei:div', namespaces=ns)
    pp_divs = tree.xpath('./tei:text/tei:body/tei:p', namespaces=ns)
    figures = tree.xpath('./tei:text/tei:body/tei:figure', namespaces=ns)
    pdf_tables = []
    for figure in figures:
        # Retrieve <table> elements within this <figure>
        # Note: Adjust the XPath expression if the structure is more complex or different
        tables = figure.xpath('./tei:table', namespaces=ns)
        if len(tables) > 0:
            pdf_tables.append(process_xml_table(tables[0]))
    extract['tables'] = pdf_tables
    sections = []
    max_section_len = 0
    for element in body_divs:
        head_text = element.xpath('./tei:head//text()', namespaces=ns)
        all_text = element.xpath('.//text()')
        # Combine text nodes into a single string
        combined_text = ''.join(all_text)
        if len(combined_text) > max_section_len:
            max_section_len = len(combined_text)
        #print(f"Section text:\n{combined_text}\n")
        if len(combined_text) > 7:
            sections.append(combined_text)
    extract["sections"] = sections
    print(f'title: {title_text}')
    print(f"Abstract: {len(abstract_text)} chars, Section count: {len(body_divs)}, tables: {len(pdf_tables)}, max_section_len: {max_section_len}")
    return extract

if __name__ == '__main__':
    extract = parse_pdf(pdf_filepath)
    print(f'{json.dumps(extract, indent=2)}\n{len(json.dumps(extract, indent=2))}')

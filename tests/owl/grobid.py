import requests
from lxml import etree
url = "http://192.168.1.160:8070/api/processFulltextDocument"
pdf_filepaths  = ["./arxiv/papers/2311.17970.pdf"                  ]

headers = {"Response-Type": "application/xml"}#, "Content-Type": "multipart/form-data"}
# Define the namespace map to handle TEI namespaces
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

for pdf_filepath in pdf_filepaths:
    files= {'input': open(pdf_filepath, 'rb')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        with open('test.tei.xml', 'w') as t:
            t.write(response.text)
    else:
        print(f'grobid error {response.status_code}, {response.text}')
        continue
    xml_content = response.text
    # Parse the XML
    tree = etree.fromstring(xml_content.encode('utf-8'))
    # Extract title
    title = tree.xpath('.//tei:titleStmt/tei:title[@type="main"]', namespaces=ns)
    title_text = title[0].text if title else 'Title not found'
    
    # Extract authors
    authors = tree.xpath('.//tei:teiHeader//tei:fileDesc//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:persName', namespaces=ns)
    authors_list = [' '.join([name.text for name in author.xpath('.//tei:forename | .//tei:surname', namespaces=ns)]) for author in authors]
    
    # Extract abstract
    abstract = tree.xpath('.//tei:profileDesc/tei:abstract//text()', namespaces=ns)
    abstract_text = ''.join(abstract).strip()
    
    # Extract major section titles
    # Note: Adjust the XPath based on the actual TEI structure of your document
    section_titles = tree.xpath('./tei:text/tei:body/tei:div/tei:head', namespaces=ns)
    titles_list = [title.text for title in section_titles]
    body_divs = tree.xpath('./tei:text/tei:body/tei:div', namespaces=ns)
    # Print extracted information
    print("Title:", title_text)
    print("Authors:", ', '.join(authors_list))
    print("Abstract:", len(abstract_text))
    print("Section titles:", len(titles_list))
    print("body divs:", len(titles_list))
    for element in body_divs:
        all_text = element.xpath('.//text()')
        # Combine text nodes into a single string
        combined_text = ''.join(all_text)
        print("Section text:", len(combined_text))
        

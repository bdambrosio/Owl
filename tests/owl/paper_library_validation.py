import os, sys
import numpy as np
import pandas as pd
import faiss
from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex

# load library
directory = './arxiv/'
papers_dir = os.path.join(os.curdir, "arxiv", "papers")
paper_library_filepath = "./arxiv/paper_library.parquet"
paper_index_filepath = "./arxiv/paper_index_w_idmap.faiss"
section_library_filepath = "./arxiv/section_library.parquet"
section_index_filepath = "./arxiv/section_index_w_idmap.faiss"
paper_library_df = pd.read_parquet(paper_library_filepath)
print('loaded paper_library_df')
paper_indexIDMap = faiss.read_index(paper_index_filepath)
print(f"loaded '{paper_index_filepath}'")
section_indexIDMap = faiss.read_index(section_index_filepath)
print(f"loaded '{section_index_filepath}'")
section_library_df = pd.read_parquet(section_library_filepath)
print(f"loaded '{section_library_filepath}'\n  keys: {section_library_df.keys()}")

papers = 0
paper_has_sections = 0

for i, paper in paper_library_df.iterrows():
    papers += 1
    sections = section_library_df[section_library_df['paper_id'].astype(str) == str(paper['faiss_id'])]
    if sections is not None and len(sections) > 0:
        paper_has_sections += len(sections)
    #print(f"{paper['faiss_id']}, {len(sections)}")

sections = 0
section_has_paper = 0
sections_dropped = 0
for i, section in section_library_df.iterrows():
    sections +=1
    paper = paper_library_df[paper_library_df['faiss_id'].astype(str) == str(section['paper_id'])]
    if paper is not None and len(paper) > 0:
        section_has_paper += 1
    else:
        section_ids_to_mask = np.array([section['faiss_id']], dtype=np.int64)
        section_indexIDMap.remove_ids(section_ids_to_mask)
        section_library_df.drop(section.name, inplace=True)
        sections_dropped += 1

faiss.write_index(section_indexIDMap, section_index_filepath)
section_library_df.to_parquet(section_library_filepath)
print(f'papers {papers} w sections {paper_has_sections}')
print(f'sections {sections} w paper {section_has_paper} dropped {sections_dropped}')

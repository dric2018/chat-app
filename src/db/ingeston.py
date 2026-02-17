from config import CFG
from db import ElectionDB

import os
import os.path as osp

import pandas as pd
import pdfplumber

if __name__=="__main__":
    db_agent = ElectionDB()

    docs_path =  osp.join(CFG.DATA_DIR, os.listdir(CFG.DATA_DIR)[0])

    db_agent.populate_db(
        db_path=CFG.DB_PATH,
        data_path=docs_path
    )
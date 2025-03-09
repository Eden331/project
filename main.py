# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:05:51 2025

@author: 97252
"""

from prep_data import main_prep as mp
from modal import main_modal as tm  # תיקון רווחים מיותרים

def main():
    # לשנות כונן אם צריך!!!!!!!!!  
    zip_path = r"C:\Eden's project\archive(2).zip"
    output_dir = r"C:\Eden's Project\unzipped"
    destination_folder = r"C:\Eden's Project\processed_data"
    
    # לשנות את הנתיב !!!  
    train_dir = r"C:\Eden's project\unzipped\Training"
    test_dir = r"C:\Eden's project\unzipped\Testing"
    
    # קריאה לפונקציות
  #  mp(zip_path, output_dir,destination_folder)
    train_dir = r"C:\Eden's Project\processed_data\test"
    test_dir = r"C:\Eden's Project\processed_data\train"
    val_dir=r"C:\Eden's Project\processed_data\val"
    tm(train_dir, test_dir, val_dir)

if __name__ == "__main__":  # תיקון רווחים מיותרים
    main()

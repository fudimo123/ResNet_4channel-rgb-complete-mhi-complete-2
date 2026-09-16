from casme2_data_parser import CASME2DataParser
import os

DATA_ROOT = os.path.abspath('../data')
ANNOTATION_FILE = os.path.join(DATA_ROOT, 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx')

parser = CASME2DataParser(ANNOTATION_FILE, DATA_ROOT)
distribution = parser.get_emotion_distribution(three_class=True)
print("3-class Emotion distribution in the dataset:")
print(distribution)
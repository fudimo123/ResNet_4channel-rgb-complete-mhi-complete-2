import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNO = os.path.join(BASE, 'data', 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx')
DYN = os.path.join(os.path.dirname(__file__), 'dynamic_data')

def main():
    df = pd.read_excel(ANNO)
    df = df[df['Estimated Emotion'] == 'others']
    missing = []
    for _, r in df.iterrows():
        subject = int(r['Subject'])
        seq = r['Filename']
        out = os.path.join(DYN, f'sub{subject:02d}', f's{subject}_{seq}.jpg')
        if not os.path.exists(out):
            missing.append(out)
    print('others total:', len(df), 'exists:', len(df) - len(missing), 'missing:', len(missing))
    if missing:
        print('example missing:', missing[:5])

if __name__ == '__main__':
    main()


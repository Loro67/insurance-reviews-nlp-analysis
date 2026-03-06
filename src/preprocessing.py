from pathlib import Path
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "Traduction_avis_clients"
def read_dataset(n:int=35):
    dataset_list = []
    for i in range(1,n + 1):
        try:
            file_name = f"avis_{i}_traduit.xlsx"
            dataset_list.append(pd.read_excel(DATA_PATH / file_name))
        except Exception as e:
            print(f"Error when reading the {file_name} ->{e}")
    return dataset_list

def fusion_dataset(dataset_list: list[pd.DataFrame])-> pd.DataFrame:
    try:
        df_final = pd.concat(dataset_list, ignore_index=True)
        return df_final  
    except Exception as e:
        print(f"Error  : {e}")
        return pd.DataFrame()

WHITELIST = ""
def spelling_correction(text: str) -> str:
    # 1. Minuscules & Nettoyage ponctuation
    # 2. Correction orthographique (ex: 'assurence' -> 'assurance')
    # 3. Retrait des stopwords (ex: 'le', 'un')
    clean_text = ""
    return clean_text
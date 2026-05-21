import pandas as pd
import numpy as np
import os
import glob
import re

def processar_csvs():
    pasta_atual = os.getcwd()
    arquivos_csv = glob.glob(os.path.join(pasta_atual, "*.csv"))

    for arquivo in arquivos_csv:
        print(f"Processando: {os.path.basename(arquivo)}")
        nome_base = os.path.basename(arquivo)
        Vn = re.search(r'^[^_]*_[^_]*_([^C]*)C', nome_base)
        Vn = int(Vn.group(1))
        cal_parametros(arquivo, Vn)


def cal_parametros(file, Vn):
    df = pd.read_csv(file)

    df["Zc [Ω/km]"] = np.round(1e3*np.sqrt(df['x1 [Ω/km]']/df['b1 [μS/km]']), 3)
    df["Pn [MW]"] = np.round(Vn**2/(df['Zc [Ω/km]']),3)
    df["Ir [A]"] = 1e3*np.round(df['Pn [MW]']/(np.sqrt(3)*Vn),3)

    df.to_csv(file)

processar_csvs()
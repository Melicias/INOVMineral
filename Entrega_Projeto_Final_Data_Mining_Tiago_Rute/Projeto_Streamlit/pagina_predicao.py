import streamlit as st
import pickle
import sklearn
import numpy as np
from explorar_dados import carrega_dados 
from sklearn.preprocessing import StandardScaler

def carrega_modelo():
    modelo_carregado = pickle.load(open('RNA_res .sav', 'rb'))
    return modelo_carregado

def carrega_padronizador():
    padronizador = pickle.load(open('padronizador.sav', 'rb'))
    return padronizador

def descodifica_predicao(prev):
    pred = prev.item()
    descodificador = {
        0 : 'Máquina OK!',
        1 : 'Máquina com falha com causa não especificada',
        2 : 'Máquina com falha de Desgaste da Ferramenta',
        3 : 'Máquina com falha de Dissipacão de Calor',
        4 : 'Máquina com falha de Potência de Energia',
        5 : 'Máquina com falha de Fratura por Fadiga',
        6 : 'Máquina com falha Aleatória I',    # com indicacão de falha de máquina
        7 : 'Máquina com falha de Desgaste da Ferramenta e Fratura por Fadiga',
        8 : 'Máquina com falha de Dissipacão de Calor e Fratura por Fadiga',
        9 : 'Máquina com falha de Potência de Energia e Fratura por Fadiga',
        10: 'Máquina com falha de Desgaste da Ferramenta, de Potência de Energia e Fratura por Fadiga',
        11: 'Máquina com falha Aleatória II',   # sem indicacão de falha de máquina
        12: 'Máquina com falha de Desgaste da Ferramenta e falha Aleatória'
        }
    return descodificador.get(pred, "Erro de descodificacão! Valor não encontrado")

# forma de inserir dados e pôr dados por defeito
def mostra_pagina_previsao():
    modelo = carrega_modelo()
    padronizador = carrega_padronizador()
    #data = carrega_dados()

    st.markdown("## Modelo de Predicão de Falhas de uma Máquina Industrial")
    st.markdown("Projeto elaborado no âmbito do Projeto da Disciplina de *Data Mining* (2021/22) do Mestrado de Ciência de Dados da ESTG - Leiria")
    st.markdown(
        ''' 
        Elaborado por: 
        :--
        Tiago Ribeiro
        Rute Fontelas

        ---

        ### Modelo de Predicão
        ''')
    st.markdown("Insira os valores das variáveis de entrada em baixo.")

    # variáveis de entrada
    niveis_qualidade = ("Alta", "Média", "Baixa")
    qualidade = st.radio("1. Nível de qualidade da peça em producão", niveis_qualidade)
    temp_ar = st.slider("2. Temperatura do ar - Região tipíca de operacão de 22 a 33 °C", min_value = 20, max_value = 45, value = 35)
    temp_proc = st.slider("3. Temperatura do processo - Região tipíca de operacão de 32 a 42 °C", min_value = 20, max_value = 45, value = 40)
    velocidade_rot = st.slider("4. Velocidade de rotação - Região tipíca de operacão de 1250 a 2000 [rpm]", min_value = 1000, max_value = 2800, value = 2000)
    binario = st.slider("5. Binário - Região tipíca de operacão de 10 a 70 [Nm]", min_value = 0, max_value = 100, value = 50)
    # desgaste da ferramenta
    
    # codifica variável de qualidade
    if qualidade == 'Alta':
        qual_cod = 2
    elif qualidade == 'Média':
        qual_cod = 1
    else:
        qual_cod = 0

    # reset = st.button("Repõe valores iniciais")
    # if reset == True:
    #     st.write()
    #     niveis_qualidade.
    #     temp_ar = 35
    #     temp_proc = 40
    #     velocidade_rot = 2000
    #     binario = 50
    #     st.subheader("RESET")
    # quando de clica no botão, valor TRUE; caso contrário é falso
    botao = st.button("Calcula Estado da Máquina")
    if botao == True:
        # entrada = data[['quality', 'air_temp', 'process_temp', 'rotational_speed', 'torque']]
        entrada = np.array([[qual_cod, temp_ar, temp_proc, velocidade_rot, binario]])
        entrada_pad = padronizador.transform(entrada)
        prev = modelo.predict(entrada_pad)
        print(type(prev))
        prev_descod = descodifica_predicao(prev.astype(int))
        st.subheader(str(prev_descod))





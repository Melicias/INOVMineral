import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
from scipy import stats

# guarda data de forma a não estar sempre aexecutar esta porcão de código
@st.cache
def carrega_dados():
    data = pd.read_csv("Proj2_ai4i2020.csv")
    # Renomeia nome das variáveis para melhor compreensão e para eliminar espacos
    data.columns = ['udi', 'product_id', 'quality', 'air_temp', 'process_temp', 
                'rotational_speed', 'torque', 'tool_wear', 'machine_failed', 
                'tool_wear_failure', 'heat_dissipation_failure', 'power_failure', 
                'overstrain_failure', 'random_failure']
    # Converte graus Kelvin para Celsius nas variáveis de Temperatura
    data["air_temp"] = data["air_temp"].subtract(272,15)
    data["process_temp"] = data["process_temp"].subtract(272,15)
    return data

def mostra_pagina_exploracao():
    st.markdown("## Exploracão do Conjunto de Dados")
    st.markdown("""Este trabalho foi baseado num conjunto de dados sintético que reflecte a manutenção preditiva real encontrada na 
    indústria.Para mais informacões acerca deste conjunto de dados consultem o link que se segue: 
    [AI4I2020 Predictive Maintenance](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).""")
    st.write(""" #### Tabela de Amostra do Conjunto de Dados """)
    st.write(""" **Variáveis de entrada:** 'quality', 'air_temp', 'process_temp', 
                'rotational_speed', 'torque'e 'tool_wear'  """)
    st.write(""" **Variáveis de saída ou objetivo:** machine_failed', 
                'tool_wear_failure', 'heat_dissipation_failure', 'power_failure', 
                'overstrain_failure', 'random_failure'  """)                   
    st.dataframe(data.head(50))
    #--------------------------------------------------------------
    st.write(""" #### Análise Gráfica da Qualidade dos Produtos e Falhas""")
    # Análise Gráfica de Variáveis Categóricas
    st.markdown(
        """ O gráfico de barras do canto superior esquerdo, relativo ao estado de operação da máquina, 
        indica que os dados relativos às situações de falha são de apenas 3,4% de um total de 10000 ocorrências, 
        o que corresponde a um total de 339 entradas com falha na máquina. Examinando o gráfico imediatamente à direita, 
        conclui-se que a maioria das falhas ocorreu devido a falha de dissipação de calor, com 32,4% do total de falhas, 
        o que corresponde a 115 ocorrências. Com valores próximos, as falhas de fratura por fadiga e falha de potência elétrica, 
        com 98 e 95 ocorrências respetivamente, formam um segundo grupo de causas de falha. 
        No gráfico do canto inferior esquerdo podemos observar que a proporção de Produtos de baixa qualidade é de 60%, 
        de média qualidade 30%, e as de alta qualidade representam apenas 10% dos produtos produzidos. 
        Esta observação levou-nos a traçar um quarto gráfico que nos indicasse se a qualidade dos produtos está relacionada 
        com a quantidade e tipo de falhas. O gráfico do canto inferior direito, indica-nos que as peças de qualidade inferior 
        apresentam uma taxa de falha ligeiramente superior às de qualidade média e superior, talvez devido ao facto das falhas 
        de fratura por fadiga terem o limite menor para produtos de qualidade inferior.
        """)


    fig6, ax = plt.subplots(2, 2, figsize=(15,10))
    ax[0,0].set_title('Machine Status')
    graph1 = ax[0,0].barh(["No Failure","Failed"], data["machine_failed"].value_counts(), color=['tab:blue', 'tab:orange'])
    ax[0,0].grid(linestyle = '--', linewidth = 0.5)

    # Escreve percentagem no gráfico de barras
    percentage = []
    for i in range(0, 2):
        percentage.append((sum(data["machine_failed"] == i)/len(data.machine_failed))*100)
    elem = 0
    for i in graph1:
        width = i.get_width()
        ax[0,0].text(x = width + .20, y = i.get_y() + i.get_height() / 2, s = str(round(percentage[elem],1))+'%')
        elem += 1
    ax[0,0].invert_yaxis()
    ax[0,0].set_xlim(0, 10500)
    # Seleciona dados para quando máquina tem falha
    ax[0,1].set_title('Types of Machine Failures')
    failure_categories = ['heat_dissipation_failure','overstrain_failure', 'power_failure',
                        'tool_wear_failure', 'random_failure']
    failure_values = data.loc[:, failure_categories].sum()
    graph2 = ax[0,1].barh(failure_categories, failure_values, color=['tab:blue', 'tab:orange','tab:green','tab:red'])
    ax[0,1].invert_yaxis()

    # Escreve percentagem no gráfico de barras
    percentage = []
    for i in range(failure_values.shape[0]):
        percentage.append((failure_values[i]/failure_values.sum())*100)
        elem = 0
    for i in graph2:
        width = i.get_width()
        ax[0,1].text(x = width + .20, y = i.get_y() + i.get_height() / 2, s = str(round(percentage[elem],1))+'%')
        elem += 1
    ax[0,1].grid(linestyle = '--', linewidth = 0.5)
    ax[0,1].set_xlim(0, 130)
    ax[1,0].set_title('Product Quality Level')
    graph3 = ax[1,0].barh(["Low", "Medium", "High"], data['quality'].value_counts(), color=['red', 'orange', 'green'])
    ax[1,0].grid(linestyle = '--', linewidth = 0.5)
    ax[1,0].invert_yaxis()
    ax[1,0].set_xlim(0, 6700)
    # Escreve percentagem no gráfico de barras
    percentage = []
    for i in range(data['quality'].value_counts().shape[0]):
        percentage.append((data['quality'].value_counts()[i]/data['quality'].value_counts().sum())*100)
        elem = 0
    for i in graph3:
        width = i.get_width()
        ax[1,0].text(x = width + .20, y = i.get_y() + i.get_height() / 2, s = str(round(percentage[elem],1))+'%')
        elem += 1
    # Gráfico de barras empilhadas
    ax[1,1].set_title('Failure Proportion by Product Quality')
    ax[1,1].grid(linestyle = '--', linewidth = 0.5)
    labels = ["Low", "Medium", "High"]
    # Queries para tipos de falha por categoria de qualidade
    tt_fail_qy = data.query('machine_failed == 1')['quality'].value_counts()
    ht_fail_qy = data.query('machine_failed == 1' and 'heat_dissipation_failure == 1')['quality'].value_counts()
    ov_fail_qy = data.query('machine_failed == 1' and 'overstrain_failure == 1')['quality'].value_counts()
    pw_fail_qy = data.query('machine_failed == 1' and 'power_failure == 1')['quality'].value_counts()
    tl_fail_qy = data.query('machine_failed == 1' and 'tool_wear_failure == 1')['quality'].value_counts()
    rd_fail_qy = data.query('random_failure == 1')['quality'].value_counts()
    tt_fail_qy = (ht_fail_qy + ov_fail_qy + pw_fail_qy + tl_fail_qy + rd_fail_qy).sort_values(ascending=False)
    #graph4 = ax[1,1].barh(labels,data.query('machine_failed == 1')['quality'].value_counts(), label = "Total Failures")
    ax[1,1].barh(labels, ht_fail_qy, label = "Heat Dissipation Failures")
    ax[1,1].barh(labels, ov_fail_qy, left = ht_fail_qy, label = "Overstrain Failures")
    ax[1,1].barh(labels, pw_fail_qy, left = ht_fail_qy + ov_fail_qy, label = "Power Failures")
    ax[1,1].barh(labels, tl_fail_qy, left = ht_fail_qy + ov_fail_qy + pw_fail_qy, label = "Tool Wear Failures")
    ax[1,1].barh(labels, rd_fail_qy, left = ht_fail_qy + ov_fail_qy + pw_fail_qy + tl_fail_qy, label = "Random Failures")
    ax[1,1].invert_yaxis()
    ax[1,1].set_xlim(0, 290)
    ax[1,1].legend()
    # Escreve percentagem no gráfico de barras
    percentage = []
    for i in range(data['quality'].value_counts().shape[0]):
        percentage.append((tt_fail_qy[i]/tt_fail_qy.sum())*100)
        elem = 0
    for i in range(data['quality'].value_counts().shape[0]):
        ax[1,1].text(x = tt_fail_qy[i] + 2, y = i, s = str(round(percentage[elem],1))+'%')
        elem += 1
    plt.tight_layout()
    st.pyplot(fig6)

    #--------------------------------------------------------------

    st.write(""" #### Gráficos da Temperatura do Ar e do Processo de Fabrico """)

    st.markdown(
        """ Observando os histogramas e caixas-de-bigodes das variáveis de temperatura do ar e do processo, 
        ambas parecem ter uma distribuição aproximada à distribuição normal com aparente simetria. 
        """)

    fig1, ax = plt.subplots(2, 2, figsize=(12,10))
    # Histograma
    bins = 40
    ax[0,0].set_title('Air and Process Temperature Histogram Observations')
    ax[0,0].grid(linestyle = '--', linewidth = 0.5)
    ax[0,0].set(xlabel='Temperature [Celsius]', ylabel='Frequency')
    # Air
    ax[0,0].hist(data.air_temp, bins, label = 'Air Temp.', edgecolor='black', linewidth=0.3)
    ax[0,0].hist(data.air_temp[data["machine_failed"] == 0], bins , label = 'Air Temp. OK', edgecolor='black', linewidth=0.3)
    ax[0,0].hist(data.air_temp[data["machine_failed"] == 1], bins, label = 'Air Temp Failure', edgecolor='black', linewidth=0.3)
    # Process
    ax[0,0].hist(data.process_temp, bins, label= 'Process Temp.', edgecolor='black', linewidth=0.3)
    ax[0,0].hist(data.process_temp[data["machine_failed"] == 0], bins , label= 'Process Temp. OK', edgecolor='black', linewidth=0.3)
    ax[0,0].hist(data.process_temp[data["machine_failed"] == 1], bins, label= 'Process Temp. Failure', edgecolor='black', linewidth=0.3)
    # Legenda
    ax[0,0].legend(framealpha = 0.5)

    # Ponto vermelho quando máquina falha; ponto azul quando máquina não falha
    col = np.where(data["machine_failed"] == 1,'tab:red','tab:blue')
    size = np.where(data["machine_failed"] == 1, 5, 1)

    # Gráfico de dispersão
    ax[0,1].set_title('Air and Process Temperature Observations Scatter Plot')
    ax[0,1].grid(linestyle = '--', linewidth = 0.5)
    ax[0,1].scatter(data.udi , data.process_temp, c = np.where(data["machine_failed"] == 1,'tab:red','lightyellow'), s = size)
    ax[0,1].scatter(data.udi , data.air_temp, c = np.where(data["machine_failed"] == 1,'tab:red','skyblue'), s = size)
    ax[0,1].set(xlabel='Unique ID', ylabel='Temperature [Celsius]')

    # Gráfico de dispersão
    ax[1,1].set_title('Air and Process Temperature Delta')
    ax[1,1].grid(linestyle = '--', linewidth = 0.5)
    ax[1,1].scatter(data.udi, data.process_temp - data.air_temp, c=col, s = size)
    ax[1,1].set(xlabel='Unique ID', ylabel='Temperature [Celsius]')
    #ax[1,1].axhspan(7, 8.6, color='red', alpha=0.1, linestyle = '--')
    ax[1,1].set_ylim(7.2, 12.5)

    # Boxplot
    ax[1,0].set_title('Air and Process Temperature Boxplots')
    ax[1,0].grid(linestyle = '--', linewidth = 0.5)
    ax[1,0].boxplot([data.air_temp, data.process_temp], vert = False)
    ax[1,0].set(xlabel='Temperature [Celsius]')
    #ax[1,0].set_xticks(['Air', 'Process'])
    plt.tight_layout()
    st.pyplot(fig1)

    #--------------------------------------------------------------

    st.write(""" #### Análise das Variáveis de Velocidade Rotacional e Diferenca de Temperatura do Processo e do Ar """)
    st.write(
        """ Começando a interpretacão do gráfico de dispersão tridimensional pela análise pela falha de dissipação de calor, 
        (o eixo x representando o ID da entrada, o eixo y a diferença entre a temperatura do processo 
        e a temperatura do ar e o eixo z a velocidade de rotação), quando a velocidade de rotação é menor que 1380 rpm e a 
        diferença entre a temperatura do processo e a temperatura do ar menor que 8,6 ℃, produz-se uma falha de dissipação de calor. 
        Observando o gráfico, torna-se evidente que existe de facto uma concentração de pontos de falha (a vermelho) nesta região. 
        """)
    # Gráfico de dispersão para análise de Falha de Dissipacão de Calor
    fig2 = plt.figure(figsize=(10,15))
    #fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    # Ponto de dimensões superiores se estiverem dentro da área que provoca falha de Dissipacão de Calor
    size = np.where((data["rotational_speed"] < 1380) & (data.process_temp - data.air_temp < 8.6), 5, 1)
    # Gréfico de Dispersão 
    #ax.set_title('Rotational Speed versus Temperature Delta')
    ax.scatter(data.udi, data.process_temp - data.air_temp, data.rotational_speed, c=col, s = size)
    # Plano z = 1380 rpm
    xx1, yy1 = np.meshgrid(np.linspace(-1000, 12000, 5), np.linspace(7.2, 8.6, 5))
    z1 = np.ones((5, 5))*1380
    ax.plot_surface(xx1, yy1, z1, alpha=0.5)
    # Plano y = 8.6 C
    xx, zz = np.meshgrid(np.linspace(-1000, 11000, 5), np.linspace(800, 1380, 5))
    yy = np.ones((5, 5))*8.6
    ax.plot_surface(xx, yy, zz, alpha=0.3)
    # Limites eixos
    ax.set_xlim(-1000, 10000)
    ax.set_ylim(7.2, 12.5)
    ax.set_zlim(800, 3000)
    # Perspectiva gráfico
    ax.azim = 5
    ax.elev = 0
    ax.set_xlabel('Unique ID')
    ax.set_ylabel('Air and Process Temperature Delta')
    ax.set_zlabel('Rotational Velocity [rpm]')
    #plt.tight_layout()
    st.pyplot(fig2)

    #--------------------------------------------------------------

    st.write(""" #### Análise Gráfica das Variáveis de Velocidade Rotacional, Binário e Desgaste da Ferramenta """)
    st.markdown(
        """ O histograma da variável da velocidade rotacional, este apresenta assimetria positiva. 
        Esta variável apresenta outliers superiores moderados e severos, sendo o ponto máximo 2886 rpm. 
        Observa-se, no entanto, que as falhas se concentram mais nos pontos inferiores à mediana.  
        Quanto às variáveis de Torque ou Binário e Desgaste da Ferramenta , verificamos que a distribuição 
        de frequências do Binário é aproximadamente normal e simétrica, com média e mediana aproximadamente em 40 Nm. 
        As falhas parecem ser mais frequentes para valores superiores à média. Relativamente à variável de desgaste, 
        apresenta uma distribuição que se pode caracterizar como uniforme, com um limite superior aproximadamente em 
        220 minutos. As falhas estão concentradas sobretudo nos pontos com valor de desgaste superior a 200 minutos. 
        """)
    fig3, ax = plt.subplots(3, 3, figsize=(16,12))
    # Ponto vermelho quando máquina falha; ponto azul quando máquina não falha
    col = np.where(data["machine_failed"] == 1,'tab:red','tab:blue')
    # Gráficos 'Rotational Speed'
    # Gráfico de dispersão
    ax[0,0].set_title('Rotational Speed Scatter Plot')
    ax[0,0].grid(linestyle = '--', linewidth = 0.5) 
    ax[0,0].scatter(data.udi, data.rotational_speed, c=col, s = 2)
    ax[0,0].set(xlabel='Unique ID',ylabel = 'Rotational Speed [rpm]')
    # Histograma
    bins = 40
    ax[1,0].set_title('Rotational Histogram')
    ax[1,0].grid(linestyle = '--', linewidth = 0.5)
    ax[1,0].set(xlabel='Rotational Speed [rpm]', ylabel='Frequency')
    ax[1,0].hist(data.rotational_speed, bins, label = 'Ok + Failure', edgecolor='black', linewidth = 0.3)
    ax[1,0].hist(data.rotational_speed[data["machine_failed"] == 0], bins, label = 'OK', edgecolor='black', linewidth = 0.3)
    ax[1,0].hist(data.rotational_speed[data["machine_failed"] == 1], bins, label = 'Failure', edgecolor='black', linewidth = 0.3)
    ax[1,0].legend(framealpha = 0.5)
    # Boxplot
    ax[2,0].set_title('Rotational Speed Boxplot')
    ax[2,0].grid(linestyle = '--', linewidth = 0.5)
    ax[2,0].boxplot(data.rotational_speed, vert = False)
    ax[2,0].set(xlabel='Rotational Speed [rpm]')

    # Gráficos 'Torque'
    # Gráfico de dispersão
    ax[0,1].set_title('Torque Scatter Plot')
    ax[0,1].grid(linestyle = '--', linewidth = 0.5) 
    ax[0,1].scatter(data.udi, data.torque, c=col, s = 2)
    ax[0,1].set(xlabel='Unique ID',ylabel='Torque [Nm]')
    # Histograma
    bins = 40
    ax[1,1].set_title('Torque Histogram')
    ax[1,1].grid(linestyle = '--', linewidth = 0.5)
    ax[1,1].set(xlabel='Torque [Nm]', ylabel='Frequency')
    ax[1,1].hist(data.torque, bins, label = 'Ok + Failure', edgecolor='black', linewidth=0.3)
    ax[1,1].hist(data.torque[data["machine_failed"] == 0], bins, label = 'OK', edgecolor='black', linewidth = 0.3)
    ax[1,1].hist(data.torque[data["machine_failed"] == 1], bins, label = 'Failure', edgecolor='black', linewidth = 0.3)
    ax[1,1].legend(framealpha = 0.5)
    # Boxplot
    ax[2,1].set_title('Torque Boxplot')
    ax[2,1].grid(linestyle = '--', linewidth = 0.5)
    ax[2,1].boxplot(data.torque, vert = False)
    ax[2,1].set(xlabel='Torque [Nm]')
    # Gráficos 'Tool Wear'
    # Gráfico de dispersão
    ax[0,2].set_title('Tool Wear Scatter Plot')
    ax[0,2].grid(linestyle = '--', linewidth = 0.5) 
    ax[0,2].scatter(data.udi, data.tool_wear, c=col, s = 2)
    ax[0,2].set(xlabel='Unique ID',ylabel='Tool Wear [minutes]')
    # 'Tool Wear Failure' de forma aleatória quando tem desgaste está entre 200 e 240 minutos
    ax[0,2].axhspan(200, 240, color='red', alpha=0.1)
    ax[0,2].set_ylim(-20, 260)
    # Histograma
    bins = 40
    ax[1,2].set_title('Tool Wear Histogram')
    ax[1,2].grid(linestyle = '--', linewidth = 0.5)
    ax[1,2].set(xlabel='Tool Wear [minutes]', ylabel='Frequency')
    ax[1,2].hist(data.tool_wear, bins, label = 'Ok + Failure', edgecolor='black', linewidth=0.3)
    ax[1,2].hist(data.tool_wear[data["machine_failed"] == 0], bins, label = 'OK', edgecolor='black', linewidth = 0.3)
    ax[1,2].hist(data.tool_wear[data["machine_failed"] == 1], bins, label = 'Failure', edgecolor='black', linewidth = 0.3)
    ax[1,2].legend(framealpha = 0.5)
    # Boxplot
    ax[2,2].set_title('Tool Wear Boxplot')
    ax[2,2].grid(linestyle = '--', linewidth = 0.5)
    ax[2,2].boxplot(data.tool_wear, vert = False)
    ax[2,2].set(xlabel='Tool Wear [minutes]')
    plt.tight_layout()
    st.pyplot(fig3)

    #--------------------------------------------------------------

    st.write(""" #### Análise Gráfica de Falhas de Potência Elétrica e Falhas Aleatórias""")
    st.markdown(
        """ As falhas de Potência Elétrica acontecem quando se ultrapassa os 9600 Watt e quando o valor 
        é menor que 3600 Watt. Estas duas regiões estão representadas a vermelho no gráfico de dispersão 
        da figura 10. Pela observação do gráfico apenas, parece-nos que a quase totalidade de ocorrências 
        nesta região estão a vermelho, isto é, representam falhas da máquina.
        As Falhas Aleatórias não são explicadas por nenhuma das variáveis dos sensores e, por essa razão, 
        não faz sentido traçar um gráfico que os relacione. Optou-se por traçar um gráfico da variável binária 
        random_failure. No total, a Falha Aleatória ocorre por 19 ocasiões em 10000.
         """)

    # Análise Gráfica de Falhas 
    fig4, ax = plt.subplots(2, 1, figsize=(12,12))
    # Ponto vermelho quando máquina falha; ponto azul quando máquina não falha
    col = np.where(data["machine_failed"] == 1,'tab:red','tab:blue')
    # 'Power Failure' se  3500 < Potencia [Watt] < 9000 ('Rotational Speed' e 'Torque')
    # Gráfico de dispersão
    ax[0].set_title('Power [Watt] = Rotational Speed x Torque - Scatter Plot')
    ax[0].grid(linestyle = '--', linewidth = 0.5)
    # Potencia [Watt] = Binário [Nm] x Velocidade Angular [rad/s] <=> Potencia [Watt] = Binário [Nm] x (Velocidade Rotacional [rpm] x 2π/60)
    ax[0].scatter(data.udi, data.rotational_speed * (2*np.pi/60) * data.torque, c = col, s = 2)
    ax[0].set(xlabel='Unique ID', ylabel = 'Power [Watt]')
    # Áreas de falha
    ax[0].axhspan(1500, 3500, color='red', alpha=0.1)
    ax[0].axhspan(9000, 11000, color='red', alpha=0.1)
    ax[0].set_ylim(1500, 11000)
    # 'Random Failure' - Gráfico Stem 
    ax[1].set_title('Random Failures - Stem Plot')
    ax[1].grid(linestyle = '--', linewidth = 0.5)
    ax[1].stem(data.udi, data.random_failure, use_line_collection = True)
    ax[1].set(xlabel='Unique ID')
    ax[1].set_yticks([0,1])
    ax[1].set_yticklabels(['0k', 'Failure'])
    plt.tight_layout()
    st.pyplot(fig4)

    #--------------------------------------------------------------

    st.write(""" #### Análise das Falhas de Fratura por Fadiga """)

    st.markdown(""" Quanto às falhas de Fratura por Fadiga, dependendo da qualidade do produto, 
    têm limites diferentes de falha. Para os produtos com qualidade superior (gráfico mais acima), 
    o nível a partir do qual existem falhas é 13000 min∙Nm. Se o produto for de qualidade média, o limite é 
    de 12000 min∙Nm, e para produtos de qualidade inferior, apenas 11000 min∙Nm """)
    # Análise Gráfica de Falhas (continuacão)
    fig5, ax = plt.subplots(3, 1, figsize=(12,12), sharex=True)
    # Ponto vermelho quando máquina falha; ponto azul quando máquina não falha
    col = np.where(data["machine_failed"] == 1,'tab:red','tab:blue')
    size = np.where(data["machine_failed"] == 1, 10, 2)
    # 'Overstrain Failure'
    # Para Produtos de qualidade 'High'   : há falha quando tool_wear x torque > 13000 min.Nm
    # Para Produtos de qualidade 'Medium' : há falha quando tool_wear x torque > 12000 min.Nm
    # Para Produtos de qualidade 'Low'    : há falha quando tool_wear x torque > 11000 min.Nm
    # Gráfico de dispersão
    # Qualidade "H"
    ax[0].set_title('Overstrain [min.Nm] for High Quality Products - Scatter Plot')
    ax[0].grid(linestyle = '--', linewidth = 0.5)
    # Cria dados de Overstrain para as diferentes quakidades de produto
    data_overstrain_H = data.tool_wear[data["quality"] == "H"] * data.torque[data["quality"] == "H"]
    data_overstrain_M = data.tool_wear[data["quality"] == "M"] * data.torque[data["quality"] == "M"]
    data_overstrain_L = data.tool_wear[data["quality"] == "L"] * data.torque[data["quality"] == "L"]
    # Potencia [Watt] = Binário [Nm] x Velocidade Angular [rad/s] <=> Potencia [Watt] = Binário [Nm] x (Velocidade Rotacional [rpm] x 2π/60)
    ax[0].scatter(data.udi[data["quality"] == "H"], data_overstrain_H , 
                c = col[data["quality"] == "H"],
                s = size[data["quality"] == "H"])
    ax[0].set(xlabel='Unique ID', ylabel = '[min.Nm]')
    ax[0].axhspan(13000, 15000, color='red', alpha=0.1)
    ax[0].set_ylim(-1000, 14500)
    # Qualidade "M"
    ax[1].set_title('Overstrain [min.Nm] for Medium Quality Products - Scatter Plot')
    ax[1].grid(linestyle = '--', linewidth = 0.5)
    # Potencia [Watt] = Binário [Nm] x Velocidade Angular [rad/s] <=> Potencia [Watt] = Binário [Nm] x (Velocidade Rotacional [rpm] x 2π/60)
    ax[1].scatter(data.udi[data["quality"] == "M"], data_overstrain_M , 
                c = col[data["quality"] == "M"],
                s = size[data["quality"] == "M"])
    ax[1].set(xlabel='Unique ID', ylabel = '[min.Nm]')
    ax[1].axhspan(12000, 15000, color='red', alpha=0.1)
    ax[1].set_ylim(-1000, 14500)
    # Qualidade "L"
    ax[2].set_title('Overstrain [min.Nm] for Low Quality Products - Scatter Plot')
    ax[2].grid(linestyle = '--', linewidth = 0.5)
    # Potencia [Watt] = Binário [Nm] x Velocidade Angular [rad/s] <=> Potencia [Watt] = Binário [Nm] x (Velocidade Rotacional [rpm] x 2π/60)
    ax[2].scatter(data.udi[data["quality"] == "L"], data_overstrain_L , 
                c = col[data["quality"] == "L"],
                s = size[data["quality"] == "L"])
    ax[2].set(xlabel='Unique ID', ylabel = '[min.Nm]')
    ax[2].axhspan(11000, 15000, color='red', alpha=0.1)
    ax[2].set_ylim(-1000, 14500)
    plt.tight_layout()
    st.pyplot(fig5)
    #--------------------------------------------------------------

    

data = carrega_dados()


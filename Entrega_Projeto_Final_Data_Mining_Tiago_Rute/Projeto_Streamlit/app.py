import streamlit as st
#import pickle
#import sklearn
from explorar_dados import mostra_pagina_exploracao
from pagina_predicao import mostra_pagina_previsao

pagina = st.sidebar.selectbox('Menu', ('Exploracão de Dados', 'Predicão'))

if pagina == 'Exploracão de Dados':
    mostra_pagina_exploracao()
elif pagina == 'Predicão':
    mostra_pagina_previsao()

#print('Finito')
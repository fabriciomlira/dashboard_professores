# ==========================================
#  dashboard_notas.py
# ==========================================
#  Academic performance dashboard
#  Created by Fabricio Lira on 07/05/26.
# ==========================================

import re
import pandas as pd
import numpy as np
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Monitoramento Pedagógico", page_icon="🎓", layout="wide")
st.title("🎓 Dashboard de Inteligência Pedagógica")

# ==========================================
# 2. EXTRAÇÃO DOS DADOS
# ==========================================
@st.cache_data
def carregar_e_processar_dados(arquivo):
    df_raw = pd.read_excel(arquivo, header=None)
    data = []
    mapa_colunas = {}
        
    current_serie, current_turma, current_aluno = "Indefinida", "Indefinida", "Indefinido" 

    for i in range(len(df_raw)):
        linha_original = df_raw.iloc[i].tolist()
        linha_str = [str(c).strip().upper() for c in linha_original]
        texto_linha = " ".join(linha_str)
        
        # 1. Identifica a Série
        if "SÉRIE:" in texto_linha:
            m = re.search(r'SÉRIE:\s*([^-\n]*)', texto_linha)
            if m: current_serie = m.group(1).strip()
            
        # 2. Identifica a Turma
        if "TURMA:" in texto_linha:
            m = re.search(r'TURMA:\s*([^\s,]*)', texto_linha)
            if m: current_turma = m.group(1).strip()

        # 3. Identifica as Disciplinas
        if "MATEMÁTICA" in texto_linha or "LÍNGUA PORTUGUESA" in texto_linha:
            mapa_colunas = {} 
            for idx, nome in enumerate(linha_str):
                if nome and nome not in ['NAN', 'NONE', 'ORD', 'ESTUDANTE', 'BIMESTRE', 'MÉDIA', 'SITUAÇÃO']:
                    mapa_colunas[nome] = idx
            continue

        # 4. Identifica o Aluno
        nome_na_celula = str(linha_original[1]).strip()
        if nome_na_celula and nome_na_celula.upper() not in ['NAN', 'ESTUDANTE', 'NONE', '']:
            current_aluno = nome_na_celula

        # 5. Identifica o Bimestre
        match_bim = re.search(r'([1-4])\s*(?:°|º|O)?\s*(?:BIM|BIMESTRE)', texto_linha)
        
        if match_bim and current_aluno != "Indefinido":
            periodo = f"{match_bim.group(1)}º BIM"
            
            # Extração das Notas
            for disc, col_idx in mapa_colunas.items():
                if col_idx < len(linha_original):
                    val = linha_original[col_idx]
                    try:
                        nota_limpa = str(val).replace(',', '.').strip()
                        nota = float(nota_limpa) if nota_limpa not in ['s/n', '-', '*', '**', 'nan', ''] else np.nan
                        
                        if not np.isnan(nota):
                            data.append({
                                'Série': current_serie, 'Turma': current_turma, 
                                'Aluno': current_aluno, 'Bimestre': periodo, 
                                'Disciplina': disc, 'Nota': nota
                            })
                    except: continue

    return pd.DataFrame(data)  
    
# ==========================================
# 3. INTELIGÊNCIA ARTIFICIAL
# ==========================================
def aplicar_ia(df_pivot, disciplinas_alvo):
    presentes = [d for d in disciplinas_alvo if d in df_pivot.columns]
    if len(df_pivot) < 3 or not presentes: return None, []

    matrix = df_pivot[presentes].fillna(df_pivot[presentes].mean()).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)
    
    #PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PCA1', 'PCA2'], index=presentes)
    
    # CLUSTER
    clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    
    df_pivot['Cluster'] = clusters
    df_pivot['PCA1'], df_pivot['PCA2'] = coords[:, 0], coords[:, 1]
    df_pivot['Média Geral'] = matrix.mean(axis=1)
    
    ordem = df_pivot.groupby('Cluster')['Média Geral'].mean().sort_values().index
    mapa = {ordem[0]: 'Crítico', ordem[1]: 'Atenção', ordem[2]: 'Excelente'}
    df_pivot['Perfil'] = df_pivot['Cluster'].map(mapa)
    
    return df_pivot, presentes, loadings
    
def interpretar_pca1(loadings):

    valores = loadings['PCA1']

    positivos = (valores > 0).sum()

    if positivos >= len(valores) * 0.8:
        return "O eixo PCA1 representa principalmente uma tendência geral de desempenho escolar."

    return "O eixo PCA1 representa padrões mistos de desempenho entre disciplinas."
    
def interpretar_pca2(loadings):

    top_pos = loadings['PCA2'].sort_values(ascending=False).head(3)
    top_neg = loadings['PCA2'].sort_values().head(3)

    texto = (
        f"⬆️ {', '.join(top_pos.index)}\n\n"
        f"⬇️ {', '.join(top_neg.index)}"
    )

    return texto
     
# ==========================================
# 4. INTERFACE E VISUALIZAÇÃO
# ==========================================
arquivo_upload = 'rel_desempenho_escolar_2026.xlsx'
#arquivo_upload = st.sidebar.file_uploader("📂 Carregar Excel", type=["xlsx"])

if arquivo_upload:
    df_base = carregar_e_processar_dados(arquivo_upload)
    
    if not df_base.empty:
        st.sidebar.header("🎯 Filtros")

        series_disponiveis = sorted(df_base['Série'].unique())
        sel_serie = st.sidebar.selectbox("Série:", series_disponiveis)
        
        df_serie = df_base[df_base['Série'] == sel_serie]
        disciplinas_da_serie = sorted(df_serie['Disciplina'].unique().tolist())
        
        turmas_disponiveis = sorted(df_serie['Turma'].unique())
        sel_turma = st.sidebar.selectbox("Turma:", turmas_disponiveis)
        
        bimestres_disponiveis = sorted(df_base['Bimestre'].unique())
        sel_bim = st.sidebar.selectbox("Bimestre:", bimestres_disponiveis)
        
        df_f = df_serie[(df_serie['Turma'] == sel_turma) & (df_serie['Bimestre'] == sel_bim)]
        
        if not df_f.empty:
            df_pivot = df_f.pivot_table(index=['Série', 'Turma', 'Aluno'], columns='Disciplina', values='Nota').reset_index()
            df_final, disciplinas_ia, loadings = aplicar_ia(df_pivot, disciplinas_da_serie)

            m_geral = df_f['Nota'].mean()
            m_mediana = df_f['Nota'].median()
            
            m_std = df_f['Nota'].std()
            cv_global = (m_std / m_geral) * 100 if m_geral > 0 else 0
            
            contagem_reprovacao = (df_final[disciplinas_ia] < 6.0).sum(axis=1)
            alunos_risco = df_final[contagem_reprovacao >= 3].copy()
            
            st.divider()
  
            st.subheader("📈 Indicadores Estratégicos")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Média", f"{m_geral:.1f}")
            c2.metric("Mediana", f"{m_mediana:.1f}", delta=f"{m_mediana - m_geral:.1f}", help="Indica a simetria. Mediana menor que a Média então a maioria da turma não está bem.")
            c3.metric("Coef. de Variação", f"{cv_global:.0f}%", delta=f"{20 - cv_global:.0f}%", help="Indica a dispersão. Valores acima de 20% sugerem uma turma heterogênea.")
            c4.metric("Risco Acadêmico", f"{len(alunos_risco)}", help="Quantidade de alunos com três ou mais reprovações.")
            
            st.divider()

            if df_final is not None:
                aba1, aba2 = st.tabs(["🔎 Análise da Turma", "👤 Análise por Aluno"])

                with aba1:
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.subheader("Mapa Espacial de Aprendizagem")
                        #st.caption("Alunos próximos têm perfis de aprendizagem semelhantes.")
                        
                        fig_pca = px.scatter(df_final, x='PCA1', y='PCA2', color='Perfil',
                                            hover_name='Aluno', hover_data={'PCA1': ':.1f', 'PCA2': ':.1f', 'Média Geral': ':.1f'},
                                            labels={'PCA1': 'Desempenho Geral', 'PCA2': 'Desempenho entre Disciplinas'},
                                            size='Média Geral', category_orders={'Perfil': ['Crítico', 'Atenção', 'Excelente']},
                                            color_discrete_map={'Crítico': '#F4A582', 'Atenção': '#E9C46A', 'Excelente': '#74C9D0'})
                        st.plotly_chart(fig_pca, use_container_width=True)
                    with col_r:
                        st.subheader("Perfil Médio dos Grupos")
                        df_heatmap = df_final.groupby('Perfil')[disciplinas_ia].mean().T
                        df_heatmap = df_heatmap[['Crítico', 'Atenção', 'Excelente']]
                        fig_heat = px.imshow(df_heatmap, color_continuous_scale='RdYlGn', text_auto=".1f", aspect="auto")
                        st.plotly_chart(fig_heat, use_container_width=True)
                        
                    # Interpretação automática
                    #st.info(interpretar_pca1(loadings))
                    st.info(interpretar_pca2(loadings))
                        
                with aba2:                 
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        aluno_sel = st.selectbox("Selecione um aluno:", sorted(df_final['Aluno'].unique()))
                        d_aluno = df_final[df_final['Aluno'] == aluno_sel]
                        st.info(f"**Perfil:** {d_aluno['Perfil'].values[0]}")
                        media_aluno = d_aluno['Média Geral'].values[0]
                        st.write(f"**Média Geral:** {media_aluno:.1f}")
                        
                        notas_aluno = d_aluno[disciplinas_ia].iloc[0]
                        abaixo_seis = notas_aluno[notas_aluno < 6.0].index.tolist()
                        
                        if not abaixo_seis:
                            st.success("🥳 Nenhuma matéria abaixo de 6.0 pontos!")
                        else:
                            lista_vertical = "\n".join([f"- {materia} ( {notas_aluno[materia]:.1f} )" for materia in abaixo_seis])
                            st.error(f"⚠️ **Notas abaixo de 6.0 pontos:**\n\n{lista_vertical}")
                    with col_r:
                        medias_turma = df_final[disciplinas_ia].mean()
                        mapa_areas = {
                            'MATEMÁTICA': 'EXATAS', 'CIÊNCIAS': 'EXATAS', 'FÍSICA': 'EXATAS', 'QUÍMICA': 'EXATAS', 'BIOLOGIA': 'EXATAS',
                            'LÍNGUA PORTUGUESA': 'LINGUAGENS','LÍNGUA PORTUGUESA E SUAS LITERATURAS': 'LINGUAGENS', 'LÍNGUA INGLESA': 'LINGUAGENS', 'ARTE': 'LINGUAGENS', 'ARTES': 'LINGUAGENS', 'EDUCAÇÃO FÍSICA': 'LINGUAGENS',
                            'ENSINO RELIGIOSO': 'HUMANAS', 'HISTÓRIA': 'HUMANAS', 'GEOGRAFIA': 'HUMANAS', 'SOCIOLOGIA': 'HUMANAS', 'FILOSOFIA': 'HUMANAS'
                        }
                        
                        ordem_areas = ['LINGUAGENS', 'HUMANAS', 'EXATAS', 'OUTROS']
                        
                        disciplinas_ord = sorted(disciplinas_ia, key=lambda d: (ordem_areas.index(mapa_areas.get(d, 'OUTROS')), d))
                        notas_ord = [notas_aluno[d] for d in disciplinas_ord]
                        medias_ord = [medias_turma[d] for d in disciplinas_ord]
            
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(r=notas_ord + [notas_ord[0]], theta=disciplinas_ord + [disciplinas_ord[0]], fill='toself', name='Nota do Aluno', line_color='#3498db'))
                        fig_radar.add_trace(go.Scatterpolar(r=medias_ord + [medias_ord[0]], theta=disciplinas_ord + [disciplinas_ord[0]], name='Média da Turma', line=dict(color='black', dash='dash')))
                        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="center", x=0.5))
                        st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Sem dados para os filtros selecionados.")
else:
    st.info("Por favor, faça o upload do arquivo Excel para iniciar a análise.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO

@st.cache_data

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Formato de archivo no soportado. Use CSV o Excel.")
        return None
    return df

def calculate_correlations(df):
    return df.corr()

def simple_regression(df, x_var, y_var):
    model = LinearRegression()
    X = df[[x_var]]
    y = df[y_var]
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    return model.coef_[0], model.intercept_, r2, y_pred

def weighted_multiple_regression(df, x_vars, y_var):
    model = LinearRegression()
    X = df[x_vars]
    y = df[y_var]
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    return model.coef_, model.intercept_, r2, y_pred

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

st.title("Modelo de Regresión para Pronóstico")

uploaded_file = st.file_uploader("Cargar archivo CSV o Excel", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())

        if st.checkbox("Transponer datos si es necesario"):
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df[1:]

        st.subheader("Correlaciones")
        corr = calculate_correlations(df.select_dtypes(include=[np.number]))
        st.dataframe(corr)

        st.subheader("Regresión Simple")
        x_var = st.selectbox("Variable independiente (X)", df.columns)
        y_var = st.selectbox("Variable dependiente (Y)", df.columns)

        if st.button("Calcular Regresión Simple"):
            coef, intercept, r2, y_pred = simple_regression(df, x_var, y_var)
            st.write(f"Coeficiente (β): {coef:.4f}")
            st.write(f"Intersección (α): {intercept:.4f}")
            st.write(f"R²: {r2:.4f}")
            df['Pronóstico Simple'] = y_pred
            st.dataframe(df[[x_var, y_var, 'Pronóstico Simple']])

        st.subheader("Regresión Múltiple Ponderada")
        x_vars = st.multiselect("Seleccionar variables independientes (X)", df.columns.tolist(), default=df.columns[:-1])
        y_var_multi = st.selectbox("Variable dependiente (Y)", df.columns)

        if st.button("Calcular Regresión Múltiple"):
            coefs, intercept, r2_multi, y_pred_multi = weighted_multiple_regression(df, x_vars, y_var_multi)
            st.write(f"Intersección (α): {intercept:.4f}")
            st.write(f"R²: {r2_multi:.4f}")
            for var, coef in zip(x_vars, coefs):
                st.write(f"Coeficiente para {var}: {coef:.4f}")
            df['Pronóstico Múltiple'] = y_pred_multi
            st.dataframe(df[x_vars + [y_var_multi, 'Pronóstico Múltiple']])

        st.subheader("Descargar resultados")
        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="Descargar Excel con resultados",
            data=excel_data,
            file_name="resultados_regresion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

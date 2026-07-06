# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Detectar automáticamente la carpeta donde está guardado este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración de rutas absolutas
archivo_original = os.path.join(BASE_DIR, "Online Retail.xlsx")
archivo_limpio = os.path.join(BASE_DIR, "online_retail_clean.csv")

def backend():
    """Función de extracción, limpieza y transformación de datos."""
    if not os.path.exists(archivo_original):
        raise FileNotFoundError(f"No se encontró el archivo: {archivo_original}")
    
    print("="*50)
    print("PROCESANDO Y LIMPIANDO EL DATASET")
    print("="*50)
    
    # Leer archivo original
    df = pd.read_excel(archivo_original)
    print(f"Dimensiones originales: {df.shape}")
    
    # Limpieza básica
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    
    # Filtros de consistencia operativa
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    
    # Tratamiento de fechas
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month_name()
    df["MonthNumber"] = df["InvoiceDate"].dt.month
    
    # Métrica de negocio
    df["Total"] = df["Quantity"] * df["UnitPrice"]
    
    # Escalado de datos
    scaler = StandardScaler()
    columnas_escalar = ["Quantity", "UnitPrice", "Total"]
    escaladas = scaler.fit_transform(df[columnas_escalar])
    
    df["Quantity_scaled"] = escaladas[:, 0]
    df["UnitPrice_scaled"] = escaladas[:, 1]
    df["Total_scaled"] = escaladas[:, 2]
    
    # Guardar archivo procesado
    df.to_csv(archivo_limpio, index=False)
    print(f"Archivo limpio generado: {archivo_limpio}\n")

def generar_reportes_locales(df):
    """Genera las matrices estadísticas y guarda reportes parciales en CSV."""
    print("="*50)
    print("GENERANDO REPORTES LOCALES Y MATRIZ")
    print("="*50)
    
    # Agrupaciones y guardado de archivos requeridos
    ventas_pais = df.groupby("Country")["Total"].sum().sort_values(ascending=False).head(10).reset_index()
    ventas_pais.to_csv("ventas_pais.csv", index=False)
    
    productos = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
    productos.to_csv("productos.csv", index=False)
    
    ventas_mes = df.groupby(["MonthNumber", "Month"])["Total"].sum().reset_index().sort_values("MonthNumber")
    ventas_mes.to_csv("ventas_mes.csv", index=False)
    
    # Matriz de Correlación Estática (Matplotlib)
    correlacion = df[['Quantity', 'UnitPrice', 'Total']].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(correlacion, cmap='Blues', interpolation='nearest')
    plt.xticks(range(len(correlacion.columns)), correlacion.columns)
    plt.yticks(range(len(correlacion.columns)), correlacion.columns)
    plt.colorbar(label="Correlación")
    plt.title("Matriz de Correlación")
    
    for i in range(len(correlacion.columns)):
        for j in range(len(correlacion.columns)):
            plt.text(j, i, round(correlacion.iloc[i, j], 2), ha='center', va='center', color='black')
    
    plt.tight_layout()
    # Guardamos la gráfica para evitar interrumpir de golpe el flujo en VS Code
    plt.savefig('matriz_correlacion.png')
    print("-> Matriz guardada localmente como 'matriz_correlacion.png'")
    plt.close() # Cierra la ventana en segundo plano de manera limpia

# ==========================================
# CONFIGURACIÓN DEL DASHBOARD INTERACTIVO
# ==========================================

# Ejecutar proceso inicial si no existe el archivo limpio
#if not os.path.exists(archivo_limpio):
 #   backend()

# Carga optimizada para el Dashboard
df_raw = pd.read_csv(archivo_limpio)
df_raw = df_raw.sort_values(by="MonthNumber")

# Convertimos explícitamente a tipos nativos de Python (int) para evitar el error de serialización
meses_disponibles = [int(m) for m in sorted(df_raw["MonthNumber"].unique())]

mes_nombres = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

# Ejecución de reportes estáticos
generar_reportes_locales(df_raw)

# DataFrames optimizados en memoria para los Callbacks
df_ventas_mes = df_raw.groupby(["Country", "MonthNumber", "Month"])["Total"].sum().reset_index()
df_productos = df_raw.groupby(["Country", "MonthNumber", "Description"])["Quantity"].sum().reset_index()
df_clientes = df_raw.groupby(["Country", "MonthNumber", "CustomerID"])["Total"].sum().reset_index()

# Inicializar Dash
app = Dash(__name__)
server = app.server

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'padding': '20px'}, children=[
    html.Div([
        html.H1("Dashboard de Ventas - Online Retail", style={'textAlign': 'center', 'color': '#1a365d', 'margin': '0'}),
        html.P("Análisis de Datos y Toma de Decisiones en Computación | Grupo 1IL-133", style={'textAlign': 'center', 'color': '#4a5568', 'marginTop': '5px'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)', 'marginBottom': '25px'}),

    html.Div([
        html.Div([
            html.Label("Filtrar por País:", style={'fontWeight': 'bold', 'color': '#2d3748', 'display': 'block', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='dropdown-pais',
                options=[{'label': 'Todos los Países', 'value': 'ALL'}] + [{'label': pais, 'value': pais} for pais in sorted(df_raw['Country'].unique())],
                value='ALL',
                clearable=False
            )
        ], style={'width': '45%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Periodo Mensual:", style={'fontWeight': 'bold', 'color': '#2d3748', 'display': 'block', 'marginBottom': '8px'}),
            dcc.RangeSlider(
                id='slider-meses',
                min=min(meses_disponibles),
                max=max(meses_disponibles),
                step=1,
                value=[min(meses_disponibles), max(meses_disponibles)],
                marks={m: mes_nombres[m] for m in meses_disponibles}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)', 'marginBottom': '25px', 'overflow': 'hidden'}),

    html.Div([
        html.Div([
            html.Div([dcc.Graph(id='grafica-ventas-mes')], style={'width': '49%', 'backgroundColor': '#fff', 'borderRadius': '12px', 'padding': '10px'}),
            html.Div([dcc.Graph(id='grafica-top-productos')], style={'width': '49%', 'backgroundColor': '#fff', 'borderRadius': '12px', 'padding': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '25px'}),

        html.Div([
            html.Div([dcc.Graph(id='grafica-distribucion')], style={'width': '49%', 'backgroundColor': '#fff', 'borderRadius': '12px', 'padding': '10px'}),
            html.Div([dcc.Graph(id='grafica-top-clientes')], style={'width': '49%', 'backgroundColor': '#fff', 'borderRadius': '12px', 'padding': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ])
])

@app.callback(
    [Output('grafica-ventas-mes', 'figure'),
     Output('grafica-top-productos', 'figure'),
     Output('grafica-distribucion', 'figure'),
     Output('grafica-top-clientes', 'figure')],
    [Input('dropdown-pais', 'value'),
     Input('slider-meses', 'value')]
)
def actualizar_dashboard(pais_seleccionado, rango_meses):
    v_mes = df_ventas_mes[(df_ventas_mes['MonthNumber'] >= rango_meses[0]) & (df_ventas_mes['MonthNumber'] <= rango_meses[1])]
    p_top = df_productos[(df_productos['MonthNumber'] >= rango_meses[0]) & (df_productos['MonthNumber'] <= rango_meses[1])]
    c_top = df_clientes[(df_clientes['MonthNumber'] >= rango_meses[0]) & (df_clientes['MonthNumber'] <= rango_meses[1])].copy()

    if pais_seleccionado != 'ALL':
        v_mes = v_mes[v_mes['Country'] == pais_seleccionado]
        p_top = p_top[p_top['Country'] == pais_seleccionado]
        c_top = c_top[c_top['Country'] == pais_seleccionado]

    # Gráfica 1: Tendencia Mensual
    v_mes_agrupado = v_mes.groupby("Month")["Total"].sum().reset_index()
    v_mes_agrupado['Month'] = pd.Categorical(v_mes_agrupado['Month'], categories=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"], ordered=True)
    v_mes_agrupado = v_mes_agrupado.sort_values('Month')

    fig1 = px.line(v_mes_agrupado, x="Month", y="Total", title="Tendencia de Ventas Mensuales",
                   labels={"Month": "Mes", "Total": "Ventas (£)"}, markers=True, color_discrete_sequence=['#3182ce'])
    fig1.update_layout(template='plotly_white')

    # Gráfica 2: Top 10 Productos
    p_agrupado = p_top.groupby("Description")["Quantity"].sum().nlargest(10).reset_index()
    fig2 = px.bar(p_agrupado, x="Quantity", y="Description", orientation="h", title="Top 10 Productos Más Vendidos",
                  labels={"Quantity": "Cantidad", "Description": "Producto"}, color="Quantity", color_continuous_scale="Blues")
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_white')

    # Gráfica 3: Distribución del Valor
    if not c_top.empty:
        umbral = c_top["Total"].quantile(0.95)
        df_hist = c_top[c_top["Total"] <= umbral]
    else:
        df_hist = c_top

    fig3 = px.histogram(df_hist, x="Total", nbins=30, title="Distribución del Valor por Cliente (95% Datos)",
                        labels={"Total": "Monto Acumulado (£)", "count": "Frecuencia"}, color_discrete_sequence=['#48bb78'])
    fig3.update_layout(template='plotly_white')

    # Gráfica 4: Top 10 Clientes
    c_top['CustomerID_Str'] = c_top['CustomerID'].astype(int).astype(str)
    c_agrupado = c_top.groupby("CustomerID_Str")["Total"].sum().nlargest(10).reset_index()
    fig4 = px.bar(c_agrupado, x="CustomerID_Str", y="Total", title="Top 10 Clientes con Mayor Volumen de Compra",
                  labels={"CustomerID_Str": "ID Cliente", "Total": "Total (£)"}, color_discrete_sequence=['#ed8936'])
    fig4.update_layout(template='plotly_white')

    return fig1, fig2, fig3, fig4

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8050)
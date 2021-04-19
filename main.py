import pandas as pd
import streamlit as st

from c2ga import Cocomo2

st.title('main.py')
st.write('Optimasi Metode COCOMO II Menggunakan Algoritma Genetika')

st.sidebar.header("Parameters")

dataset_select = st.sidebar.selectbox(
    "Select dataset",
    (
        "turkish.csv",
        # "nasa93.csv",
    )
)

a = st.sidebar.number_input('A', value=2.94)
b = st.sidebar.number_input('B', value=0.91)

source = pd.read_csv(dataset_select, sep=";")
c = Cocomo2(source, a=a, b=b)

st.header("Load dataset")
st.write(f'Using `{dataset_select}` dataset (change on sidebar)')
st.write(c.df)

with st.beta_expander("View effort multiplier columns"):
    em = c.em_values
    st.write(f"{len(c.em_cols)} column(s)", ", ".join(c.em_cols))
    st.write(c.em_values)

st.header("Effort multiplier (EM)")
st.latex(r"""EM = \prod_{i=1}^{17} SF_i""")
st.write(c.effort_multipliers)

st.header("Objective function (estimated effort, EE)")
st.latex(r"""EE = F(A,B)""")
st.latex(r"""F(A,B) = A \times Size^B \times EM""")
st.write(c.estimated_efforts)

st.header("Magnitude relative error (MRE)")
st.latex(r"""MRE = \frac{|AE-EE|}{EE} \times 100\%""")
st.write(c.magnitude_relative_error * 100)

st.header("Mean magnitude relative error (MMRE)")
st.latex(r"""MMRE = \frac{1}{N} \sum_{x=1}^{N} MRE""")
st.dataframe([c.mean_magnitude_relative_error * 100])

import pandas as pd
import streamlit as st

from c2ga import Cocomo2

# -----------------------------------------------------------------------------

st.set_page_config(
    page_title='Optimasi Metode COCOMO II Menggunakan Algoritma Genetika',
    page_icon='👨‍💻',
)

# -----------------------------------------------------------------------------

st.sidebar.write('''
## Parameters
''')

dataset_select = st.sidebar.selectbox(
    'Select dataset',
    [
        'turkish.csv'
        # 'nasa93.csv',
    ]
)

a = st.sidebar.number_input('A', value=2.94)
b = st.sidebar.number_input('B', value=0.91)

source = pd.read_csv(dataset_select, sep=';')
c = Cocomo2(source, a, b)

# -----------------------------------------------------------------------------

with st.beta_container():
    st.write(f'''
    # Optimasi Metode COCOMO II Menggunakan Algoritma Genetika
    
    <https://griko.dev/undergrad-py> · <https://github.com/grikomsn/undergrad-thesis>
    
    ## Load dataset
    
    Using `{dataset_select}` dataset (change on sidebar)
    ''')

    st.write(pd.DataFrame(c.data))

# -----------------------------------------------------------------------------

with st.beta_container():
    st.write('''
    ## Compute EM, EE, MRE, MMRE
    
    - estimated effort = objective function -> F(A,B)
    - SF = scale factor
    ''')

    left, right = st.beta_columns(2)

    left.write(r'''
    ### Effort multiplier (EM)
    
    $$
    EM = \prod_{i=1}^{17} SF_i
    $$
    ''')

    left.write(
        pd.DataFrame(c.effort_multipliers, columns=['EM'])
    )

    right.write(r'''
    ### Estimated effort (EE)
    
    $$
    F(A,B) = A \times Size^B \times EM
    $$
    ''')

    right.write(
        pd.DataFrame(c.estimated_efforts, columns=['F(A,B)'])
    )

# -----------------------------------------------------------------------------

with st.beta_container():
    left, right = st.beta_columns(2)

    left.write(r'''
    ### Magnitude relative error (MRE)
    
    $$
    MRE = \frac{|AE-EE|}{EE} \times 100\%
    $$
    ''')

    left.write(
        pd.DataFrame(c.mre * 100, columns=['MRE'])
    )

    right.write(r'''
    ### Mean magnitude relative error (MMRE)
    
    $$
    MMRE = \frac{1}{N} \sum_{x=1}^{N} MRE
    $$
    ''')

    right.write(
        pd.DataFrame([c.mmre * 100], columns=['MMRE'])
    )

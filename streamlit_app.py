import pandas as pd
import streamlit as st

from c2ga import Cocomo2

TITLE = 'Optimasi Metode COCOMO II Menggunakan Algoritma Genetika'


def main():
    st.set_page_config(page_title=TITLE, page_icon='👨‍💻')
    st.title(TITLE)
    st.write('<https://griko.dev/undergrad-py> · <https://github.com/grikomsn/undergrad-thesis> \n ---')
    run_app()


@st.cache()
def create_cocomo(source: str, a: float, b: float):
    data = pd.read_csv(source, sep=';')
    return Cocomo2(data, a, b)


def run_app():
    a: float
    b: float
    dataset_select: str

    with st.sidebar.beta_container():
        st.write('## Parameters')
        dataset_select = st.selectbox(
            'Select dataset',
            [
                'turkish.csv'
                # 'nasa93.csv',
            ]
        )
        a = st.number_input('A', value=2.94)
        b = st.number_input('B', value=0.91)

    c = create_cocomo(dataset_select, a, b)

    st.write(f'## Load dataset \n Currently using `{dataset_select}` dataset (change on sidebar)')
    st.dataframe(c.data)

    '---'

    st.write('## Compute EM, EE, MRE')
    with st.beta_container():
        left, right = st.beta_columns(2)

        left.write('### Effort multiplier (EM)')
        left.latex(r'EM = \prod_{i=1}^{17} SF_i')
        left.write('> SF = scale factor')

        left.write('### Estimated effort (EE)')
        left.latex(r'F(A,B) = A \times Size^B \times EM')
        left.write('> EE = objective function -> F(A,B)')

        left.write('### Magnitude relative error (MRE)')
        left.latex(r'MRE = \frac{|AE-EE|}{EE} \times 100\%')

        right.write('### Results')
        right.dataframe({
            'EM': c.effort_multipliers,
            'F(A,B)': c.estimated_efforts,
            'MRE': c.mre * 100,
        })

        st.write('---')

    st.write('## Compute MMRE')
    with st.beta_container():
        left, right = st.beta_columns([2, 1])

        left.write('### Mean magnitude relative error (MMRE)')
        left.latex(r'MMRE = \frac{1}{N} \sum_{x=1}^{N} MRE')

        right.write('### Results')
        right.write(pd.DataFrame({'MMRE': [c.mmre * 100]}))


if __name__ == '__main__':
    main()

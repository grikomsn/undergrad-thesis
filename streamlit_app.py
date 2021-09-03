import numpy as np
import pandas as pd
import streamlit as st

from undergrad_thesis.cocomo2 import Cocomo2
from undergrad_thesis.ga import GeneticAlgorithm

st.set_page_config(page_title='Optimasi Metode COCOMO II Menggunakan Algoritma Genetika', page_icon='👨‍💻')

'''
# Optimasi Metode COCOMO II Menggunakan Algoritma Genetika
<https://griko.dev/undergrad-py> · <https://github.com/grikomsn/undergrad-thesis>
'''


# load turkish.csv cocomo data
@st.cache()
def load_csv_data():
    return pd.read_csv('turkish.csv', sep=';')


csv_data = load_csv_data()

# initialize cocomo object
c = Cocomo2(data=csv_data, a=2.94, b=0.91)

'''
## 1. Dataset COCOMO II

Sourced from `turkish.csv`
'''
st.write(csv_data)

'## 2. Compute EM, EE, MRE, MMRE'
with st.container():
    left, right = st.columns(2)

    left.write(r'''
    ### Effort multiplier (EM)
    
    $$
    EM = \prod_{i=1}^{17} SF_i
    $$
    
    *) SF = scale factor
    
    ### Estimated effort (EE)
    
    $$
    EE = A \times Size^B \times EM
    $$
    ''')

    right.write(r'''
    ### Magnitude relative error (MRE)
    
    $$
    MRE = \frac{|AE-EE|}{EE}
    $$
    
    ### Mean magnitude relative error (MMRE)
    
    $$
    MMRE = \frac{1}{N} \sum_{x=1}^{N} MRE
    $$
    ''')

    left, right = st.columns([2, 1])

    left.write(pd.DataFrame({
        'EM': c.effort_multipliers,
        'EE': c.estimated_efforts(),
        'MRE': c.magnitude_relative_error(),
        'MRE %': c.magnitude_relative_error() * 100,
    }))

    right.write(pd.DataFrame({
        'MMRE': [c.mean_magnitude_relative_error()],
        'MMRE %': [c.mean_magnitude_relative_error() * 100],
    }))

'---'

'## 3. Genetic Algorithm'

test_indiv = [2.94, 0.91]
genes_size = len(test_indiv)

with st.sidebar:
    population_size = st.slider(
        label='Population size',
        min_value=2,
        max_value=100,
        step=1,
        value=50,
    )

    crossover_rate = st.slider(
        label='Crossover rate',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.7,
    )

    mutation_rate = st.slider(
        label='Mutation rate',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.1,
    )

ga = GeneticAlgorithm(
    seed_data=test_indiv,
    population_size=population_size,
    generations=100,
    crossover_probability=crossover_rate,
    mutation_probability=mutation_rate,
    elitism=True,
    maximise_fitness=False
)


def create_individual(_):
    return list(np.random.uniform(size=genes_size, low=0.0, high=1.0))


ga.create_individual = create_individual


def objective_function(indiv):
    [a, b] = indiv
    return c.estimated_efforts(a, b)


def fitness_function(indiv, _data):
    ee = objective_function(indiv)
    return 1.0 / (1.0 + ee.mean())


ga.fitness_function = fitness_function

with st.container():
    left, right = st.columns(2)

    left.write(r'''
    ### Objective function
    
    $$
    F(A,B) = \frac{1}{N} \sum_{i=1}^{N} EE_i
    $$
    
    *) objective fn = avg. estimated effort per project
    
    ### Fitness function
    
    $$
    F(x) = \frac{1}{1+f(x)}
    $$
    $$
    f(x) = F(A,B)
    $$
    ''')

    right.write('''
    ### Test fitness function w/ objective function
    ''')

    right.write(pd.DataFrame({
        'Fitness': [fitness_function(test_indiv, [])],
        'EE': [1.0 / (1.0 + objective_function(test_indiv).mean())],
    }))


def splice_generation(generation):
    spliced = {
        'Fitness': [],
        'A': [],
        'B': [],
    }
    for chrmsm in generation:
        [a, b] = chrmsm.genes
        spliced['Fitness'].append(chrmsm.fitness)
        spliced['A'].append(a)
        spliced['B'].append(b)
    return spliced


'---'


def fyi_best(chrmsm):
    best_fitness, [best_a, best_b] = chrmsm

    return pd.DataFrame({
        # 'Best fitness': [best_fitness],
        'Best fitness %': [best_fitness * 100],
        'Best A': [best_a],
        'Best B': [best_b],
    })


r'''
### Run first generation

Generated population with constraint $0.0 \ge x_i \ge 1.0$
'''

ga.create_first_generation()

with st.container():
    'Generation #001'
    left, right = st.columns(2)
    left.write(pd.DataFrame(splice_generation(ga.current_generation)))
    right.write(fyi_best(ga.best_individual()))

progress = st.progress(1)

with st.expander(f'Generation 2-{ga.generations}'):
    for i in range(1, ga.generations):
        progress.progress(i)
        ga.create_new_population()
        ga.calculate_population_fitness()
        ga.rank_population()
        with st.container():
            f'Generation #{(i + 1):03}'
            left, right = st.columns(2)
            left.write(pd.DataFrame(splice_generation(ga.current_generation)))
            right.write(fyi_best(ga.best_individual()))
    progress.progress(100)
    progress.empty()

'---'

with st.container():
    final_fitness, [final_a, final_b] = ga.best_individual()

    f'''
    ## 4. Conclusion
    
    Result from {ga.generations} generations: 
    
    - Best fitness: `{final_fitness}`
    - Best fitness %: `{final_fitness * 100} %`
    - Best A: `{final_a}`
    - Best B: `{final_b}`
    '''

# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd
import streamlit as st
from pandas import DataFrame, Series
from pandas.api.extensions import ExtensionArray
from pandas.io.parsers import TextFileReader
from pyeasyga.pyeasyga import GeneticAlgorithm

UnifiedDataFrame = Union[DataFrame, ExtensionArray, Series, TextFileReader]

st.set_page_config(page_title='Optimasi Metode COCOMO II Menggunakan Algoritma Genetika', page_icon='👨‍💻')

'''
# Optimasi Metode COCOMO II Menggunakan Algoritma Genetika
<https://griko.dev/undergrad-py> · <https://github.com/grikomsn/undergrad-thesis>
'''


# main cocomo ii instance
class Cocomo2:
    data: UnifiedDataFrame

    a: float
    b: float

    em_cols = [
        'RELY', 'DATA', 'CPLX', 'RUSE', 'DOCU', 'TIME', 'STOR', 'PVOL', 'ACAP', 'PCAP', 'PCON', 'APEX', 'PLEX', 'LTEX',
        'TOOL', 'SITE', 'SCED'
    ]

    def __init__(self, data: UnifiedDataFrame, a: float = None, b: float = None):
        self.data = data
        self.a = a
        self.b = b

    @property
    def em_values(self) -> UnifiedDataFrame:
        return self.data[self.em_cols]

    @property
    def locs(self) -> UnifiedDataFrame:
        return self.data['LOC']

    @property
    def actual_efforts(self) -> UnifiedDataFrame:
        return self.data['AE']

    @property
    def effort_multipliers(self) -> UnifiedDataFrame:
        return self.em_values.prod(axis=1)

    def estimated_efforts(self, a: float = None, b: float = None) -> UnifiedDataFrame:
        if a is None:
            if self.a is None:
                raise Exception('parameter a is empty')
            a = self.a
        if b is None:
            if self.b is None:
                raise Exception('parameter b is empty')
            b = self.b

        em = self.effort_multipliers
        size = self.locs / 1000
        return a * size.pow(b) * em

    def magnitude_relative_error(self, ee: UnifiedDataFrame = None) -> UnifiedDataFrame:
        ae = self.actual_efforts

        if ee is None:
            ee = self.estimated_efforts()

        return (ae - ee).abs() / ae

    def mean_magnitude_relative_error(self, mre: UnifiedDataFrame = None):
        if mre is None:
            mre = self.magnitude_relative_error()

        return mre.mean()


# load turkish.csv cocomo data
csv_data = pd.read_csv('turkish.csv', sep=';')

# initialize cocomo object
c = Cocomo2(data=csv_data, a=2.94, b=0.91)

'## Dataset COCOMO II turkish.csv'
st.write(csv_data)

'## Compute EM, EE, MRE'
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
    
    ### Magnitude relative error (MRE)
    
    $$
    MRE = \frac{|AE-EE|}{EE}
    $$
    
    ### Mean magnitude relative error (MMRE)
    
    $$
    MMRE = \frac{1}{N} \sum_{x=1}^{N} MRE
    $$
    ''')

    right.write(pd.DataFrame({
        'EM': c.effort_multipliers,
        'EE': c.estimated_efforts(),
        'MRE': c.magnitude_relative_error(),
        'MRE %': c.magnitude_relative_error() * 100,
    }))

    right.write(pd.DataFrame(pd.DataFrame({
        'MMRE': [c.mean_magnitude_relative_error()],
        'MMRE %': [c.mean_magnitude_relative_error() * 100],
    })))

'---'

'## Genetic Algorithm'

test_indiv = [2.94, 0.91]
genes_size = len(test_indiv)

with st.sidebar:
    population_size = st.slider(
        label="Population size",
        min_value=2,
        max_value=100,
        step=1,
        value=50,
    )

    crossover_rate = st.slider(
        label="Crossover rate",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.7,
    )

    mutation_rate = st.slider(
        label="Mutation rate",
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
    f(x) = F(A,B)
    $$
    ''')

    right.write('''
    ### Test fitness function w/ objective function
    ''')

    right.write(pd.DataFrame({
        "fitness fn result": [fitness_function(test_indiv, [])],
        "ee result": [1.0 / (1.0 + objective_function(test_indiv).mean())],
    }))


def splice_generation(generation):
    spliced = {
        "Fitness": [],
        "A": [],
        "B": [],
    }
    for chrmsm in generation:
        [a, b] = chrmsm.genes
        spliced["Fitness"].append(chrmsm.fitness)
        spliced["A"].append(a)
        spliced["B"].append(b)
    return spliced


'---'

r'''
### Run first generation

Generated population with constraint $0.0 \ge x_i \ge 1.0$
'''

ga.create_first_generation()

'Generation #1'
st.write(pd.DataFrame(splice_generation(ga.current_generation)))

for i in range(1, ga.generations):
    ga.create_new_population()
    ga.calculate_population_fitness()
    ga.rank_population()

    f'Generation #{i + 1}'
    st.write(pd.DataFrame(splice_generation(ga.current_generation)))

(best_fitness, best_genes) = ga.best_individual()

f'''
**Best individual**
- fitness: {best_fitness}
- genes: {best_genes} 
'''

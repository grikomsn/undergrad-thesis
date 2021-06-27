from c2ga.typings import UnifiedDataFrame


class Cocomo2:
    data: UnifiedDataFrame

    a: float
    b: float

    em_cols = [
        'RELY', 'DATA', 'CPLX', 'RUSE', 'DOCU', 'TIME', 'STOR', 'PVOL', 'ACAP', 'PCAP', 'PCON', 'APEX', 'PLEX', 'LTEX',
        'TOOL', 'SITE', 'SCED'
    ]

    def __init__(self, data: UnifiedDataFrame, a: float, b: float):
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

    @property
    def estimated_efforts(self) -> UnifiedDataFrame:
        em = self.effort_multipliers
        size = self.locs / 1000
        return self.a * size.pow(self.b) * em

    @property
    def magnitude_relative_error(self) -> UnifiedDataFrame:
        ae = self.actual_efforts
        ee = self.estimated_efforts
        return (ae - ee).abs() / ae

    @property
    def mre(self):
        return self.magnitude_relative_error

    @property
    def mean_magnitude_relative_error(self):
        return self.magnitude_relative_error.mean()

    @property
    def mmre(self):
        return self.mean_magnitude_relative_error

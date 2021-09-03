from undergrad_thesis.utils import UnifiedDataFrame


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

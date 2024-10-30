import polars as pl
from typing import Literal
import numpy as np
import pandas as pd
import glob

# 如果 PATH_PARQUETS 和 cs 没有定义，您需要定义它们
# 例如：
# PATH_PARQUETS = '/path/to/your/parquets'
# cs = your_defined_cs

class DatasetConstructor:
    def __init__(self, mode: Literal['train', 'test']):
        self.mode = mode
        self.path = PATH_PARQUETS / mode

    @staticmethod
    def reduce_memory_usage_pl(df):
        """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
            Original pandas version of this function: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 """
        print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
        Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
        Numeric_Float_types = [pl.Float32, pl.Float64]
        for col in df.columns:
            try:
                col_type = df[col].dtype
                if col_type == pl.Categorical:
                    continue
                c_min = df[col].min()
                c_max = df[col].max()
                if col_type in Numeric_Int_types:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df = df.with_columns(df[col].cast(pl.Int8))
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df = df.with_columns(df[col].cast(pl.Int16))
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df = df.with_columns(df[col].cast(pl.Int32))
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df = df.with_columns(df[col].cast(pl.Int64))
                elif col_type in Numeric_Float_types:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df = df.with_columns(df[col].cast(pl.Float32))
                    else:
                        pass
                # elif col_type == pl.Utf8:
                #     df = df.with_columns(df[col].cast(pl.Categorical))
                else:
                    pass
            except:
                pass
        print(f"Memory usage of dataframe became {round(df.estimated_size('mb'), 2)} MB")
        return df

    @staticmethod
    def detect_datetime_cols(df):
        return df.select_dtypes(object).apply(lambda x: pd.to_datetime(x, errors='ignore'), axis=0).select_dtypes(
            np.datetime64).columns.tolist()

    def _to_pandas(self, df):
        df = df.to_pandas().set_index('case_id')
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def merge_static(self, df):
        df_static = (
            pl.concat(
                [pl.scan_parquet(p, low_memory=True) for p in glob.glob(str(self.path / f"{self.mode}_static_0_*"))],
                how="vertical_relaxed", )
            .with_columns(
                [
                    (pl.col(col).cast(pl.String).str.to_date(strict=False))
                    for col in [
                    'datefirstoffer_1144D',
                    'datelastinstal40dpd_247D',
                    'datelastunpaid_3546854D',
                    'dtlastpmtallstes_4499206D',
                    'firstclxcampaign_1125D',
                    'firstdatedue_489D',
                    'lastactivateddate_801D',
                    'lastapplicationdate_877D',
                    'lastapprdate_640D',
                    'lastdelinqdate_224D',
                    'lastrejectdate_50D',
                    'lastrepayingdate_696D',
                    'maxdpdinstldate_3546855D',
                    'payvacationpostpone_4187118D',
                    'validfrom_1069D'
                ]
                ] + [
                    (pl.col(col).cast(pl.String).cast(pl.Categorical))
                    for col in [
                        'bankacctype_710L', 'cardtype_51L', 'credtype_322L',
                        'disbursementtype_67L', 'equalitydataagreement_891L',
                        'equalityempfrom_62L', 'inittransactioncode_186L',
                        'isbidproductrequest_292L', 'isdebitcard_729L',
                        'lastapprcommoditycat_1041M', 'lastapprcommoditytypec_5251766M',
                        'lastcancelreason_561M', 'lastrejectcommoditycat_161M',
                        'lastrejectcommodtypec_5251769M', 'lastrejectreason_759M',
                        'lastrejectreasonclient_4145040M', 'lastst_736L', 'opencred_647L',
                        'paytype1st_925L', 'paytype_783L', 'previouscontdistrict_112M',
                        'twobodfilling_608L', 'typesuite_864L'
                    ]
                ]
            )
        )
        return df.join(df_static, how="left", on="case_id")

    def merge_static_cb(self, df):
        df_static_cb = (
            pl.scan_parquet(self.path / f"{self.mode}_static_cb_0.parquet", low_memory=True)
            .with_columns(
                [
                    (pl.col(col).cast(pl.String).str.to_date(strict=False))
                    for col in [
                    'assignmentdate_238D',
                    'assignmentdate_4527235D',
                    'assignmentdate_4955616D',
                    'birthdate_574D',
                    'dateofbirth_337D',
                    'dateofbirth_342D',
                    'responsedate_1012D',
                    'responsedate_4527233D',
                    'responsedate_4917613D'
                ]
                ] + [
                    (pl.col(col).cast(pl.String).cast(pl.Categorical))
                    for col in [
                        'description_5085714M', 'education_1103M', 'education_88M',
                        'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L',
                        'riskassesment_302T'
                    ]
                ]
            )
        )
        return df.join(df_static_cb, how="left", on="case_id")

    def load(self):
        df = pl.scan_parquet(self.path / f"{self.mode}_base.parquet", low_memory=True).with_columns(
            pl.col("date_decision").str.to_date()
        )
        # Depth=0
        df = self.merge_static(df)
        df = self.merge_static_cb(df)

        df = (
            df
            .with_columns(
                pl.col(pl.Float64).cast(pl.Float32),
                pl.col(pl.Int64).cast(pl.Int32),
            )
        )
        df = df.select(~cs.date())

        # Drop categorical large-dimension columns
        df = df.drop([
            'lastapprcommoditytypec_5251766M',
            'previouscontdistrict_112M',
            'district_544M',
            'profession_152M',
            'name_4527232M',
            'name_4917606M',
            'employername_160M',
            'classificationofcontr_400M',
            'financialinstitution_382M',
            'contaddr_district_15M',
            'contaddr_zipcode_807M',
            'empladdr_district_926M',
            'empladdr_zipcode_114M',
            'registaddr_district_1083M',
            'registaddr_zipcode_184M',
            'addres_district_368M',
            'addres_zip_823M'])
        df = df.collect()
        df = self.reduce_memory_usage_pl(df)
        df = self._to_pandas(df)
        return df
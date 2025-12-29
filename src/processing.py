import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self, drop_cols = None):
        self.drop_cols = drop_cols if drop_cols else []
        self.cat_cols = []
        self.num_cols = []
        self.drop_miss_cols = []
        self.miss_flag = []
        self.num_medians = {}
        self.fill_na = {}
        self.outliers_col = {}
        self.mapping_dict = {}
        self.ordinal_cols = []
        self.cate_cols = []

    def fit(self, df):
        df = df.copy()
        # Drop unnecessary columns
        df = df.drop(columns=self.drop_cols, errors='ignore')

        # Drop columns with more than 50% missing values
        miss_count = df.isnull().sum()
        miss_percent = miss_count / len(df)
        self.drop_miss_cols = miss_percent[miss_percent > 0.5].index.tolist()
        df = df.drop(columns=self.drop_miss_cols)

        # Identify categorical and numerical columns
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Flag columns with missing values
        self.miss_flag = [col for col in df.columns if df[col].isnull().any()]

        # Calculate medians for numerical columns
        if "Neighborhood" in df.columns:
            for col in self.num_cols:
                self.num_medians[col] = df.groupby("Neighborhood")[col].median().to_dict()
        else:
            for col in self.num_cols:
                self.num_medians[col] = df[col].median()

        # Calculate modes for categorical columns
        for col in self.cat_cols:
            mode = df[col].mode()
            self.fill_na[col] = mode.iloc[0] if len(mode) > 0 else 'Missing'

        # Determine outlier thresholds for numerical columns
        for col in self.num_cols:
            n_unique = df[col].nunique()
            zero_ratio = (df[col] == 0).mean()

            if n_unique <= 10:
                continue
            if zero_ratio >= 0.3 and df[col].quantile(0.25) == 0:
                continue

            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            self.outliers_col[col] = (q01, q99) 

        # Define mappings for ordinal categorical columns
        quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        lotshape_map = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
        landslope_map = {"Gtl": 3, "Mod": 2, "Sev": 1}
        bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1}
        bsmtfin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1}
        functional_map = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1}
        electrical_map = {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP": 2, "Mix": 1}
        paveddrive_map = {"Y": 3,"P": 2,"N": 1}

        self.mapping_dict = {
            "LotShape": lotshape_map,
            "LandSlope": landslope_map,
            "ExterQual": quality_map,
            "ExterCond": quality_map,
            "BsmtQual": quality_map,
            "BsmtCond": quality_map,
            "HeatingQC": quality_map,
            "KitchenQual": quality_map,
            "FireplaceQu": quality_map,
            "GarageQual": quality_map,
            "GarageCond": quality_map,
            "BsmtExposure": bsmt_exposure_map,
            "BsmtFinType1": bsmtfin_map,
            "BsmtFinType2": bsmtfin_map,
            "Functional": functional_map,
            "Electrical": electrical_map,
            "PavedDrive": paveddrive_map
        }

        self.ordinal_cols = list(self.mapping_dict.keys())
        self.cate_cols = [col for col in self.cat_cols if col not in self.ordinal_cols]

        df_tmp = df.copy()
        df_tmp = self.feature_engineering(df)
        df_dum = pd.get_dummies(df_tmp, columns=self.cate_cols, drop_first=True)
        self.dummy_cols = df_dum.columns
        return self
    
    def feature_engineering(self, df):
        new_features = {
            "HouseAge": df["YrSold"] - df["YearBuilt"],
            "YearsSinceRemod": df["YrSold"] - df["YearRemodAdd"],
            "Remodeled": (df["YearBuilt"] != df["YearRemodAdd"]).astype(int),
            "HasLotFrontage": (~df["LotFrontage"].isna()).astype(int),
            "TotalArea": (df["GrLivArea"] + df["TotalBsmtSF"] + df["GarageArea"] + df["1stFlrSF"] + df["2ndFlrSF"]),
            "LogGrLivArea": np.log1p(df["GrLivArea"]),
            "LogTotalArea": np.log1p(df["GrLivArea"] + df["TotalBsmtSF"] + df["GarageArea"] + df["1stFlrSF"] + df["2ndFlrSF"]),
            "LogLotArea": np.log1p(df["LotArea"]),
        }

        new_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_df], axis=1)

    
    def transform(self, df):
        df = df.copy()
        # Drop unnecessary columns
        df = df.drop(columns=self.drop_cols, errors='ignore')

        # Drop columns with more than 50% missing values
        df = df.drop(columns=self.drop_miss_cols)

        # Flag columns with missing values
        for col in self.miss_flag:
            if col in df.columns:
                df[f'{col}_was_missing'] = df[col].isnull().astype(int)

        # Fill missing values for numerical columns
        for col in self.num_cols:
            if col in df.columns:
                if "Neighborhood" in df.columns and isinstance(self.num_medians[col], dict):
                    mapped = df["Neighborhood"].map(self.num_medians[col])
                    df[col] = df[col].fillna(mapped)
                    df[col] = df[col].fillna(np.median(list(self.num_medians[col].values())))
                else:
                    df[col] = df[col].fillna(self.num_medians[col])

        # Fill missing values for categorical columns
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.fill_na[col])

        # Cap outliers for numerical columns
        for col, (q01, q99) in self.outliers_col.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=q01, upper=q99)

        # Map ordinal categorical columns
        for col, mapping in self.mapping_dict.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1)

        df = self.feature_engineering(df)

        df = pd.get_dummies(df, columns=self.cate_cols, drop_first=True)
        df = df.reindex(columns=self.dummy_cols, fill_value=0)

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
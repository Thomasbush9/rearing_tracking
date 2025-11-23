import numpy as np
import pandas as pd



class Helper:
    def __init__(self, project):
        self.project = project
        #extract feautes_names 
        self.features_names = project.get_feature_set()

    def extract_features_df(self, session) -> pd.DataFrame:
        """Extract the features from the session and return a dataframe with selected features.
        
        Handles special feature names:
        - Direct matches: If feature exists in df.columns, uses it directly
        - Displacement: If feature ends with '_displacement' or '_disp', computes 
          displacement of base feature (e.g., 'height_displacement' -> diff of 'height')
        - Mean features: If feature not found, attempts to match L/R variants and 
          computes mean (e.g., 'forepaws_tail_distance' -> mean of 'forepawL_tail_distance' 
          and 'forepawR_tail_distance')
        
        Example:
            If config has feature_set: ["forepaws_tail_distance", "height", "height_displacement"]
            and df has columns: ["forepawL_tail_distance", "forepawR_tail_distance", "height"]
            Returns df with:
            - forepaws_tail_distance: mean(forepawL_tail_distance, forepawR_tail_distance)
            - height: direct column
            - height_displacement: diff(height) with 0 prepended
        """
        if not session.has_features():
            raise ValueError("Session has no features")
        
        features_file = session.features_file()
        if features_file.suffix == ".csv":
            df = pd.read_csv(features_file)
        elif features_file.suffix == ".xlsx":
            df = pd.read_excel(features_file)
        else:
            raise ValueError(f"Unsupported file format: {features_file.suffix}")
        
        result_dict = {}
        for feat_name in self.features_names:
            if feat_name in df.columns:
                # Fill NaN values to prevent NaN loss in training
                result_dict[feat_name] = df[feat_name].ffill().bfill().fillna(0).values
            elif feat_name.endswith("_displacement") or feat_name.endswith("_disp"):
                base_name = feat_name.replace("_displacement", "").replace("_disp", "")
                if base_name in df.columns:
                    result_dict[feat_name] = self.get_displacement_feature(df[base_name].values)
                else:
                    raise ValueError(f"Base feature '{base_name}' not found for displacement '{feat_name}'")
            else:
                matching_cols = []
                for col in df.columns:
                    if col.replace("L", "").replace("R", "") in feat_name.replace("L", "").replace("R", "") or \
                       feat_name.replace("L", "").replace("R", "") in col.replace("L", "").replace("R", ""):
                        if col != feat_name:
                            matching_cols.append(col)
                if len(matching_cols) >= 2:
                    result_dict[feat_name] = self.get_mean_features(df, matching_cols)
                else:
                    raise ValueError(f"Feature '{feat_name}' not found in dataframe and not a recognized special feature")
        
        return pd.DataFrame(result_dict)
        
    def get_displacement_feature(self, feature:np.ndarray) -> np.ndarray:
        """Get the displacement of a feature and it stack a 0 a the beginning"""
        #check shape of feature
        if feature.ndim != 1:
            raise ValueError("Feature must be 1-dimensional")
        # Fill NaN values with forward fill, then backward fill for leading NaNs
        feature_clean = pd.Series(feature).ffill().bfill().fillna(0).values
        displacement = np.diff(feature_clean)
        displacement = np.concatenate(([0], displacement))
        return displacement
    def get_mean_features(self, df:pd.DataFrame, features:list[str]) -> np.ndarray:
        """Get the mean of a list of features"""
        if not all(feature in df.columns for feature in features):
            raise ValueError("Features not found in dataframe")
        # Use skipna=True to handle NaN values
        return df[features].mean(axis=1, skipna=True).fillna(0).values
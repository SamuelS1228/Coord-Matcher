import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

EARTH_RADIUS_KM = 6371.0


def validate_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()

    # Drop duplicate columns (keep first)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        # Optional: surface which duplicates were dropped
        st.warning(f"{name}: dropping duplicated columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    required_cols = {"Lat", "Long"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {', '.join(missing)}. "
            f"Columns found: {list(df.columns)}"
        )

    # Ensure numeric Lat/Long
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["Lat", "Long"]].isna().any().any():
        bad_rows = df[df[["Lat", "Long"]].isna().any(axis=1)]
        raise ValueError(
            f"{name} contains non-numeric or missing values in Lat/Long columns. "
            f"Example bad rows:\n{bad_rows.head().to_string(index=False)}"
        )

    df = df.reset_index(drop=True)
    return df


def haversine_distance_matrix(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance between each (lat1, lon1) and each (lat2, lon2).

    lat1, lon1: arrays of shape (n,)
    lat2, lon2: arrays of shape (m,)
    Returns: distances (km) of shape (n, m)
    """
    # Convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Broadcasting to create (n, m) arrays
    dlat = lat1[:, None] - lat2[None, :]
    dlon = lon1[:, None] - lon2[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def match_nearest(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df2.empty:
        raise ValueError("List 2 is empty. Cannot match against an empty list.")

    lat1 = df1["Lat"].to_numpy()
    lon1 = df1["Long"].to_numpy()
    lat2 = df2["Lat"].to_numpy()
    lon2 = df2["Long"].to_numpy()

    distances = haversine_distance_matrix(lat1, lon1, lat2, lon2)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist_km = distances[np.arange(distances.shape[0]), nearest_idx]

    matched_df1 = df1.reset_index(drop=True).copy()
    matched_df2 = df2.reset_index(drop=True).iloc[nearest_idx].copy()

    matched_df1.rename(columns={"Lat": "Lat_1", "Long": "Long_1"}, inplace=True)
    matched_df2.rename(columns={"Lat": "Lat_2", "Long": "Long_2"}, inplace=True)

    result = pd.concat([matched_df1, matched_df2], axis=1)
    result["Distance_km"] = nearest_dist_km
    return result


def main():
    st.title("Coordinate Nearest-Match Tool")

    st.markdown(
        """Upload two CSV files:

        - **List 1**: Coordinates to be matched (Lat, Long)
        - **List 2**: Reference coordinates (Lat, Long)

        The app will match each coordinate in List 1 to the nearest coordinate in List 2 using the Haversine formula.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload List 1 (Lat/Long)", type=["csv"], key="list1")
    with col2:
        file2 = st.file_uploader("Upload List 2 (Lat/Long)", type=["csv"], key="list2")

    if file1 and file2:
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)

            df1 = validate_df(df1, "List 1")
            df2 = validate_df(df2, "List 2")

            st.subheader("Preview: List 1")
            st.dataframe(df1.head())

            st.subheader("Preview: List 2")
            st.dataframe(df2.head())

            if st.button("Run Matching"):
                with st.spinner("Matching coordinates..."):
                    result = match_nearest(df1, df2)

                st.subheader("Matched Results (List 1 -> Nearest in List 2)")
                st.dataframe(result.head(100))

                # Download
                buffer = BytesIO()
                result.to_csv(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="Download full matches as CSV",
                    data=buffer,
                    file_name="coordinate_matches.csv",
                    mime="text/csv",
                )

                st.success("Matching complete. You can scroll through the table above or download the full CSV.")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

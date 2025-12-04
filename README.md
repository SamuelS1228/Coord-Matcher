# Coordinate Nearest-Match Streamlit App (v3)

This app lets you upload two lists of coordinates and matches each coordinate in **List 1** to the nearest coordinate in **List 2** using the Haversine distance on latitude/longitude.

This version expects different column names for each list to avoid any ambiguity:

- **List 1** must have columns: `Lat1`, `Long1`
- **List 2** must have columns: `Lat2`, `Long2`

Extra columns are allowed and will be carried through the matching step.

## File Format

**List 1 example:**

```csv
Lat1,Long1
42.3601,-71.0589
40.7128,-74.0060
```

**List 2 example:**

```csv
Lat2,Long2
42.3610,-71.0570
40.7306,-73.9352
```

## How to Run Locally

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the URL that Streamlit prints in your terminal (usually http://localhost:8500).

## Sample Data

The `sample_data` folder contains two example files:

- `list1_sample.csv` (Lat1/Long1)
- `list2_sample.csv` (Lat2/Long2)

You can use these to test the app quickly.

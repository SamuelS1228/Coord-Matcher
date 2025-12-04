# Coordinate Nearest-Match Streamlit App

This app lets you upload two lists of coordinates and matches each coordinate in **List 1** to the nearest coordinate in **List 2** using the Haversine distance on latitude/longitude.

## File Format

Each list must be a CSV file with at least these columns:

- `Lat`
- `Long`

Example:

```csv
Lat,Long
42.3601,-71.0589
40.7128,-74.0060
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

- `list1_sample.csv`
- `list2_sample.csv`

You can use these to test the app quickly.

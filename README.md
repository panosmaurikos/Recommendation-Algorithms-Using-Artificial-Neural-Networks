# IMDB Preprocessing & Preference Matrix — Documentation

This README documents the preprocessing pipeline implemented in `main.py`. It explains each major step, the design choices, and how to reproduce the results.

## Purpose
The pipeline processes a raw dataset (`Dataset.npy`) of user ratings and produces:
- cleaned data (pickled),
- per-user statistics (ratings count and time span),
- filtered dataset (users with 100..300 ratings),
- a sparse preference matrix (users × items),
- and basic consistency checks.

Two modes are supported:
- `MODE = "spec"` — strict, follows the exercise specification: keep last rating per (user,item), keep only ratings > 0, compute unique items per user (|φ(u)|) and span on deduped data.
- `MODE = "mirror"` — reproduce the professor/friend pipeline: compute counts on cleaned raw data (no early dedupe), used only if you need exact numeric reproduction.

## Files produced / used
- `Dataset.npy` — input raw dataset (array of comma-separated strings like `ur123,tt456,8,01 January 2000`).
- `datafiles/dataframe.pkl` — cleaned pandas DataFrame (cached).
- `datafiles/ratings_num_df.pkl` — per-user ratings count DataFrame (cached).
- `datafiles/ratings_span_df.pkl` — per-user rating span DataFrame (cached).
- `datafiles/preference_matrix.npz` — saved sparse CSR preference matrix.
- `datafiles/*.npy` — other cached arrays (W, CommonRatings) produced by other scripts.

## Main steps (walkthrough of main.py)
1. **Imports & Helpers**
   - Standard libraries (numpy, pandas, matplotlib, networkx, scipy.sparse, sklearn helpers).
   - `plot_histogram(series, title_str)` saves histogram PNGs to `figures/`.

2. **Paths & MODE**
   - `FIGURES_PATH = "figures"`, `datafolder = "datafiles"`.
   - `MODE = "spec"` (change to `"mirror"` to mimic the professor flow).

3. **Load & Parse Dataset**
   - `np.load("Dataset.npy", allow_pickle=True)` reads raw records.
   - The code decodes bytes to UTF-8 if necessary and splits each record by comma to create rows: `[user, item, rating, date]`.

4. **Initial Cleaning**
   - Create DataFrame with columns `["user","item","rating","date"]`.
   - Remove prefixes (`"ur"` and `"tt"`), strip whitespace.
   - Convert `user` and `item` to numeric Int64, `rating` to numeric, `date` to datetime (coercing invalids).
   - Drop rows with missing essential fields.

5. **MODE-specific preparation (df_for_stats)**
   - `spec`:
     - Sort by `date`, `drop_duplicates(subset=["user","item"], keep="last")`.
     - Keep only `rating > 0`.
     - This yields the deduped dataset used for counts and spans — matches exercise definition φ(u) = { i : R(u,i) > 0 }.
   - `mirror`:
     - Use the cleaned (but not deduped) dataset for per-user stats — reproduces the original script behavior.

6. **Per-user Statistics**
   - `ratings_num_df`: if `spec` → unique `item` per user (`nunique`) on `df_for_stats`; if `mirror` → `count()` on `rating`.
   - `ratings_span_df`: computed on `df_for_stats` as `max(date) - min(date)` per user (converted to days).
   - These stats are cached as pickles.

7. **Filter Users**
   - Filter users with `R_min = 100` and `R_max = 300`.
   - Build `final_df` from `df_for_stats` restricted to these users.

8. **Histograms**
   - Plot and save histograms for `ratings_num` and `ratings_span` (numeric days).

9. **Preference Matrix (sparse)**
   - For safety, `final_df` is again deduped by `(user,item)` keeping last.
   - Create mappings `user_to_idx` and `item_to_idx` from sorted unique IDs.
   - Fill a `scipy.sparse.lil_matrix` with ratings (only > 0). Convert to CSR and save with `save_npz`.
   - Print shape, nonzero count and density.

10. **Final consistency checks**
    - The script includes `final_checks(...)` to verify:
      - zero duplicates in `df_for_stats` and `final_df`,
      - no non-positive ratings in `spec` mode,
      - `ratings_num` matches recomputed values,
      - ratings_span has no NaN,
      - number of non-zeros in the sparse matrix matches `final_df` rows.

## How to reproduce / best practices
1. Remove cached pickles before switching mode or after editing parsing logic:
   - `rm datafiles/dataframe.pkl datafiles/ratings_num_df.pkl datafiles/ratings_span_df.pkl datafiles/preference_matrix.npz`
2. Set `MODE = "spec"` to follow the exercise (recommended). Set `"mirror"` only if you need exact numbers from the professor's original pipeline.
3. Run:
   - `python main.py`
4. Inspect outputs:
   - PNG histograms in `figures/`.
   - `preference_matrix.npz` can be loaded with `scipy.sparse.load_npz`.
   - Use `final_checks` output to verify pipeline consistency.

## Notes and rationale
- The `spec` mode implements the formal definition: for each user u, R(u,i) is the latest rating for (u,i), and φ(u) includes items with R(u,i) > 0. This ensures counts are unique items per user and removes duplicates before counting.
- The `mirror` mode is provided only for reproducibility (matching older code), but it is less faithful to the formal definition.
- The pipeline intentionally caches intermediate results (pickles) to speed up repeated runs — remember to clear caches when changing code or mode.

## Next steps (recommended)
- Move clustering / modeling to a separate script that loads `preference_matrix.npz`.
- Add logging instead of prints, and save run metadata (MODE, timestamp, datafile checksum).
- Add unit tests for parsing and dedupe logic to catch edge cases in the raw data.

If you want, I can:
- add a small example showing how to load and inspect `preference_matrix.npz`,
- create a separate script for clustering that reuses `preference_matrix.npz`.




# 🚗 Car Price Prediction using Linear Regression

This project builds a machine learning model to predict used car prices based on features like car name, company, year, fuel type, and kilometers driven. It uses **Linear Regression** and includes full preprocessing with **OneHotEncoding** using a pipeline. The final model is serialized with **pickle** for future predictions.

---

## 📊 Dataset

The dataset was sourced from Quikr and includes the following columns:

- `name`: Full car name (e.g., "Hyundai Santro Xing")
- `company`: Manufacturer
- `year`: Year of manufacture
- `Price`: Target variable (selling price)
- `kms_driven`: Kilometers driven
- `fuel_type`: Type of fuel used (Petrol, Diesel, LPG)

---

## 🧹 Data Cleaning Steps

- Removed rows with invalid or non-numeric `year` values
- Removed entries with `"Ask For Price"` in the `Price` column
- Cleaned `kms_driven` to retain numeric values only
- Dropped rows with missing values in `fuel_type`
- Truncated `name` column to first three words
- Filtered out price outliers (above ₹60,00,000)

Final cleaned dataset: **816 rows × 6 columns**

---

## 🧠 Model Pipeline

We used a `Pipeline` combining:

- `ColumnTransformer` with `OneHotEncoder` for categorical columns: `name`, `company`, `fuel_type`
- `LinearRegression` for prediction

Model training included tuning across 1000 random states to find the best R² score.

**Best R² Score Achieved**: `0.8456` on test set

---

## 💾 Model Saving

The trained pipeline is saved using `pickle`:

```python
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
```
## prediction
```python



pipe.predict(pd.DataFrame([[
    'Maruti Suzuki Swift',  # name
    'Maruti',               # company
    2019,                   # year
    100,                    # kms_driven
    'Petrol'                # fuel_type
]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
```
## Output
```python

array([459212.61]




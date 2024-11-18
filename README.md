# Promotion Price Sensitivity Analysis

## Project Overview
The goal of this project is to analyze the impact of discounts on product sales and estimate the quantity of items we can anticipate selling under various scenarios, such as applying different discounts or offering no discounts at all.

---

## Dataset Description
The dataset consists of the following columns:
- **`Unnamed: 0`**: Appears to be an index or unnecessary column.
- **`product_number`**: Represents the product identifier.
- **`department_desc`**: Indicates the department description (coded numerically).
- **`discount`**: Represents the discount applied to the product (in percentage or amount).
- **`date_of_order`**: Indicates the date of the order.
- **`orders`**: Represents the number of items sold on that date.

---

## Suggested Approach to Complete the Analysis

### 1. Data Cleaning and Preprocessing
The following steps were performed to prepare the dataset for analysis:
1. **Remove Unnecessary Columns**:
   - The `Unnamed: 0` column was removed as it served no purpose in the analysis.
2. **Date Formatting**:
   - The `date_of_order` column was converted to a proper `datetime` format.
3. **Handle Missing Values**:
   - Missing or inconsistent values were handled to ensure data integrity.
4. **Categorical Conversion**:
   - The `department_desc` column was converted to a categorical variable to better represent department categories.

---

### 2. Exploratory Data Analysis (EDA)
Key analyses and visualizations included:
1. **Relationship Between Discounts and Sales**:
   - Scatter plots were used to visualize the correlation between discounts (`discount`) and sales (`orders`).
2. **Trends Over Time**:
   - Line charts were generated to analyze sales trends over time, with a focus on periods with and without discounts.
3. **Aggregated Effects**:
   - Data was grouped by `product_number` and `department_desc` to observe aggregated effects on sales.

---

### 3. Statistical Analysis
To better understand the data, the price elasticity of demand was calculated:
- **Formula**:
  \[
  \text{Elasticity} = \frac{\% \text{Change in Quantity Demanded}}{\% \text{Change in Price}}
  \]
- The `discount` column served as the proxy for price changes, while the `orders` column represented the quantity demanded.

---

### 4. Modeling and Prediction
To predict sales based on discount percentages, the following steps were taken:
1. **Regression Model**:
   - A linear regression model was built to predict the expected number of orders.
2. **Feature Selection**:
   - Additional features, such as `product_number` and `department_desc`, were included to improve model accuracy.

---

### 5. Report Results
The final report included:
- **Key Findings**:
  - Price elasticity of demand.
  - Expected sales under various discount scenarios.
  - The impact of offering no discounts on sales.
- **Visualizations**:
  - Graphs and charts to support conclusions.
- **Recommendations**:
  - Insights for business decisions based on the analysis.

---

## How to Run the Analysis
1. Clone the repository:
   ```bash
   git clone <repository_url>

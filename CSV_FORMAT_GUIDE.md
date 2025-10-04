# Sample CSV Format for Stock Price Prediction

## Required Columns

Your CSV file must contain at least these two columns:

### 1. Date
- **Format**: YYYY-MM-DD (e.g., 2025-01-15)
- **Type**: Date
- **Required**: Yes

### 2. Close
- **Format**: Decimal number (e.g., 175.43)
- **Type**: Float/Number
- **Required**: Yes
- **Description**: Closing price of the stock

## Sample CSV Structure

```csv
Date,Close
2023-01-03,125.07
2023-01-04,126.36
2023-01-05,125.02
2023-01-06,129.62
2023-01-09,130.15
2023-01-10,130.73
2023-01-11,133.49
2023-01-12,133.41
2023-01-13,134.76
...
```

## Optional Columns

While not required, you can include additional columns which will be ignored:

- **Open**: Opening price
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Volume**: Trading volume
- **Adj Close**: Adjusted closing price

## Full Example

```csv
Date,Open,High,Low,Close,Volume,Adj Close
2023-01-03,130.28,130.90,124.17,125.07,112117500,124.58
2023-01-04,126.89,128.66,125.08,126.36,89113600,125.87
2023-01-05,127.13,127.77,124.76,125.02,80962700,124.53
2023-01-06,126.01,130.29,124.89,129.62,87754700,129.11
2023-01-09,130.47,133.41,129.89,130.15,70790800,129.64
```

## Requirements

1. **Minimum Rows**: At least 100 rows recommended (more data = better predictions)
2. **Date Format**: Must be YYYY-MM-DD
3. **Numeric Values**: Close prices must be valid numbers
4. **No Missing Values**: Ensure no empty cells in Date or Close columns
5. **Chronological Order**: Dates should be in chronological order (oldest first)

## Tips for Best Results

1. **More Data is Better**: Include at least 1-2 years of historical data
2. **Consistent Intervals**: Daily data works best
3. **No Gaps**: Try to avoid missing trading days
4. **Clean Data**: Remove any rows with invalid or missing values

## Creating Your CSV

### From Excel or Google Sheets:
1. Organize your data with Date and Close columns
2. Select File > Download > CSV (.csv)
3. Save the file

### From Yahoo Finance:
1. Visit finance.yahoo.com
2. Search for your stock
3. Go to "Historical Data" tab
4. Select date range
5. Click "Download" to get CSV file

### From Other Sources:
Ensure your data matches the format above, then save as CSV.

## Example Download

You can download sample CSV files from:
- Yahoo Finance: https://finance.yahoo.com/quote/AAPL/history
- Google Finance: https://www.google.com/finance
- Alpha Vantage: https://www.alphavantage.co

## Troubleshooting

### Common Issues:

**"Invalid file type"**
- Ensure the file has .csv extension
- Don't use Excel format (.xlsx or .xls)

**"No data found"**
- Check that column names are exactly "Date" and "Close"
- Verify dates are in YYYY-MM-DD format

**"Insufficient data"**
- Ensure you have at least 60-100 rows of data
- Check that Close column has valid numeric values

**"Training failed"**
- Verify no missing values in Close column
- Ensure all dates are valid and in order

## Processing Your Upload

Once uploaded:
1. The system will validate your CSV format
2. Extract Date and Close columns
3. Train LSTM model on your data
4. Generate predictions
5. Display visualizations and metrics

---

**Note**: For best results, use clean, consistent historical data with at least 1 year of daily prices.

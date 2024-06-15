# Custom Big-Vul
This dataset contains C/C++ functions pre-processed for line-level vulnerability detection. It consists of 8,794 vulnerable and 177,736 non-vulnerable functions.

It is based on [Big-Vul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) (specifically *MSR_data_cleaned.zip* found on this page) with the following changes:
- Block and inline comments have been removed, including those marking vulnerable lines.
- Tabs have been replaced by double spaces.
- Additional vulnerable lines have been added by considering contol and data flow dependencies of lines added in the fix.
- Some incorrectly labelled functions have been removed.

The data is provided as CSV and JSON, and is sorted by vulnerability labels (i.e. vulnerable functions are at the end of the dataset).

The data consists of the following columns:
- `code`: function code
- `vul`: vulnerability label of the function (1 if vulnerable, 0 if non-ulnerable)
- `flaw_line_no`: line numbers (starting from 1) of vulnerable lines. These may include lines which are control or data dependent on lines added in the fix.
- `bigvul_id`: reference id to original Big-Vul dataset
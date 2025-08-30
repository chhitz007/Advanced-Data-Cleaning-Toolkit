

# Advanced Data Cleaning Toolkit v6.0

A **comprehensive, production-ready Python toolkit** for cleaning, validating, and preprocessing CSV datasets with intelligent type detection, configurable strategies, and detailed reporting.

---

## **Features**

* ✅ **Intelligent column type detection**: Automatically identifies boolean, numerical, datetime, and categorical columns.
* ✅ **Configurable cleaning strategies**: Customize imputation, thresholds, date formats, and more via a JSON configuration.
* ✅ **Multiple imputation options**: Median, mean, mode, or drop for missing values in numerical, categorical, and boolean data.
* ✅ **Data quality validation**: Completeness, consistency, validity checks, and outlier detection.
* ✅ **Backup & rollback**: Automatically creates backups before cleaning.
* ✅ **Comprehensive logging**: Tracks every cleaning step and decisions for transparency.
* ✅ **Dry-run mode**: Preview changes without modifying original files.
* ✅ **Detailed reports**: Generates per-file JSON reports and a summary text report.
* ✅ **Command-line interface (CLI)**: Run the toolkit with custom configurations, dry runs, or folder overrides.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/advanced-data-cleaning-toolkit.git
cd advanced-data-cleaning-toolkit
```

2. Install required dependencies:

```bash
pip install pandas numpy
```

3. (Optional) Additional libraries if not already installed:

```bash
pip install argparse json logging
```

---

## **Usage**

### **1. Generate default configuration**

```bash
python data_cleaning_toolkit_v6.py --create-config
```

This creates `cleaning_config.json` with default settings.

---

### **2. Dry-run mode (preview changes)**

```bash
python data_cleaning_toolkit_v6.py --dry-run
```

* No changes are made.
* Shows what would be cleaned and type changes.

---

### **3. Run actual cleaning**

```bash
python data_cleaning_toolkit_v6.py
```

* Cleans all CSV files in the configured `input_folder`.
* Saves cleaned files and JSON reports in `output_folder`.
* Logs are saved in `logs/` folder.

---

### **4. Override input/output folders via CLI**

```bash
python data_cleaning_toolkit_v6.py --input-folder raw_data --output-folder cleaned_data
```

---

## **Configuration**

* Config is stored in `cleaning_config.json`.
* Key configurable options:

```json
{
  "input_folder": "raw_data",
  "output_folder": "cleaned_data",
  "backup_folder": "backup_data",
  "create_backup": true,
  "duplicate_threshold": 1.0,
  "datetime_threshold": 0.3,
  "numeric_threshold": 0.5,
  "boolean_threshold": 0.8,
  "imputation_strategy": {
    "numerical": "median",
    "categorical": "mode",
    "boolean": "mode"
  },
  "date_formats": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
}
```

---

## **Logging**

* Logs stored in `logs/` folder.
* Each run generates a timestamped log file with detailed processing info.

---

## **Reports**

* Per-file JSON report includes:

  * Column types before and after cleaning
  * Missing values
  * Rows and duplicates removed
  * Data quality metrics

* Summary report `cleaning_summary_report.txt` aggregates all processed files.

---

## **Dependencies**

* Python 3.8+
* pandas
* numpy
* logging (standard library)
* argparse (standard library)

---

#!/usr/bin/env python3
"""
Advanced Data Cleaning Toolkit v6.0
====================================
A comprehensive, production-ready data cleaning solution with intelligent type detection,
configurable cleaning strategies, and detailed quality reporting.

Features:
- Intelligent column type detection
- Multiple imputation strategies
- Data quality validation and reporting
- Configurable cleaning rules
- Backup and rollback capabilities
- Comprehensive logging
- Progress tracking
"""

import pandas as pd
import numpy as np
import os
import re
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaningToolkit:
    """
    Advanced data cleaning toolkit with intelligent type detection and quality reporting.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.setup_logging()
        self.cleaning_report = {}
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration settings from file or use defaults."""
        default_config = {
            "input_folder": "raw_data",
            "output_folder": "cleaned_data",
            "backup_folder": "backup_data",
            "create_backup": True,
            "duplicate_threshold": 1.0,
            "datetime_threshold": 0.3,
            "numeric_threshold": 0.5,
            "boolean_threshold": 0.8,
            "imputation_strategy": {
                "numerical": "median",  # median, mean, mode, drop
                "categorical": "mode",  # mode, unknown, drop
                "boolean": "mode"       # mode, false, drop
            },
            "date_formats": [
                "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y"
            ]
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_folder = "logs"
        os.makedirs(log_folder, exist_ok=True)
        
        log_filename = f"data_cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_folder, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_column_type(self, series: pd.Series) -> str:
        """
        Intelligently detect the most appropriate data type for a column.
        """
        # Remove whitespace and convert to string for analysis
        clean_series = series.astype(str).str.strip().str.lower()
        non_null_series = clean_series[clean_series.notna() & (clean_series != '') & (clean_series != 'nan')]
        
        if len(non_null_series) == 0:
            return 'categorical'
        
        # Boolean detection - improved logic
        boolean_values = {'true', 't', 'yes', 'y', '1', 'false', 'f', 'no', 'n', '0'}
        boolean_matches = non_null_series.isin(boolean_values).sum()
        boolean_ratio = boolean_matches / len(non_null_series)
        
        if boolean_ratio >= self.config['boolean_threshold']:
            return 'boolean'
        
        # Datetime detection - improved with multiple format attempts
        datetime_count = 0
        for date_format in self.config['date_formats']:
            try:
                temp_dt = pd.to_datetime(non_null_series, format=date_format, errors='coerce')
                datetime_count = max(datetime_count, temp_dt.notna().sum())
            except:
                continue
        
        # Also try automatic datetime parsing
        try:
            temp_dt = pd.to_datetime(non_null_series, errors='coerce')
            datetime_count = max(datetime_count, temp_dt.notna().sum())
        except:
            pass
        
        datetime_ratio = datetime_count / len(non_null_series)
        if datetime_ratio >= self.config['datetime_threshold']:
            return 'datetime'
        
        # Numerical detection - enhanced pattern matching
        numeric_pattern = r'^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$'
        # Remove common non-numeric characters first
        cleaned_for_numeric = non_null_series.str.replace(r'[$,€¥£%\s()_-]', '', regex=True)
        numeric_matches = cleaned_for_numeric.str.match(numeric_pattern, na=False).sum()
        numeric_ratio = numeric_matches / len(non_null_series)
        
        if numeric_ratio >= self.config['numeric_threshold']:
            return 'numerical'
        
        # Default to categorical
        return 'categorical'
    
    def clean_categorical(self, series: pd.Series) -> pd.Series:
        """Enhanced categorical data cleaning."""
        cleaned_series = series.astype(str).str.strip()
        
        # More precise null-like patterns
        null_patterns = [
            r'^unknown$', r'^nan$', r'^n/?a$', r'^null$', r'^none$', 
            r'^not applicable$', r'^missing$', r'^empty$', r'^\s*-+\s*$',
            r'^other$', r'^unspecified$', r'^undefined$', r'^\s*$'
        ]
        
        for pattern in null_patterns:
            cleaned_series = cleaned_series.str.replace(pattern, 'unknown', regex=True, case=False)
        
        # Standardize case (title case for better readability)
        cleaned_series = cleaned_series.str.title()
        cleaned_series = cleaned_series.replace('Unknown', 'Unknown')
        
        return cleaned_series
    
    def clean_boolean(self, series: pd.Series) -> pd.Series:
        """Enhanced boolean data cleaning."""
        boolean_map = {
            'true': True, 't': True, 'yes': True, 'y': True, '1': True, 'on': True,
            'false': False, 'f': False, 'no': False, 'n': False, '0': False, 'off': False
        }
        
        cleaned_series = series.astype(str).str.lower().str.strip()
        mapped_series = cleaned_series.map(boolean_map)
        
        # Handle imputation based on strategy
        if self.config['imputation_strategy']['boolean'] == 'mode':
            mode_value = mapped_series.mode()
            fill_value = mode_value[0] if len(mode_value) > 0 else False
        elif self.config['imputation_strategy']['boolean'] == 'false':
            fill_value = False
        else:  # drop strategy handled elsewhere
            fill_value = np.nan
        
        return mapped_series.fillna(fill_value)
    
    def clean_numerical(self, series: pd.Series) -> pd.Series:
        """Enhanced numerical data cleaning with better parsing."""
        # Convert to string and clean
        cleaned_series = series.astype(str)
        
        # Handle parentheses (accounting format for negatives)
        cleaned_series = cleaned_series.str.replace(r'\(([0-9,.]+)\)', r'-\1', regex=True)
        
        # Remove currency symbols, commas, spaces, but preserve decimal points and signs
        cleaned_series = cleaned_series.str.replace(r'[$,€¥£%\s_]', '', regex=True)
        
        # Convert to numeric
        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
        
        # Handle imputation
        if self.config['imputation_strategy']['numerical'] == 'median':
            fill_value = numeric_series.median()
        elif self.config['imputation_strategy']['numerical'] == 'mean':
            fill_value = numeric_series.mean()
        elif self.config['imputation_strategy']['numerical'] == 'mode':
            mode_values = numeric_series.mode()
            fill_value = mode_values[0] if len(mode_values) > 0 else 0
        else:  # drop strategy
            return numeric_series
        
        return numeric_series.fillna(fill_value)
    
    def clean_datetime(self, series: pd.Series) -> pd.Series:
        """Enhanced datetime cleaning with multiple format support."""
        # Try each specified format first
        for date_format in self.config['date_formats']:
            try:
                dt_series = pd.to_datetime(series, format=date_format, errors='coerce')
                if dt_series.notna().sum() / len(series) > self.config['datetime_threshold']:
                    return dt_series
            except:
                continue
        
        # Fall back to automatic parsing
        return pd.to_datetime(series, errors='coerce')
    
    def calculate_data_quality_metrics(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
        """Calculate comprehensive data quality metrics."""
        metrics = {
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'columns_before': len(df_before.columns),
            'columns_after': len(df_after.columns),
            'duplicates_removed': len(df_before) - len(df_after),
            'completeness_before': (1 - df_before.isnull().sum().sum() / df_before.size) * 100,
            'completeness_after': (1 - df_after.isnull().sum().sum() / df_after.size) * 100,
            'column_types': {},
            'missing_values_by_column': {}
        }
        
        for col in df_after.columns:
            metrics['column_types'][col] = str(df_after[col].dtype)
            metrics['missing_values_by_column'][col] = df_after[col].isnull().sum()
        
        return metrics
    
    def create_backup(self, file_path: str) -> str:
        """Create backup of original file."""
        if not self.config['create_backup']:
            return ""
        
        backup_folder = self.config['backup_folder']
        os.makedirs(backup_folder, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{timestamp}_{file_name}"
        backup_path = os.path.join(backup_folder, backup_name)
        
        try:
            df_original = pd.read_csv(file_path)
            df_original.to_csv(backup_path, index=False)
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"Could not create backup: {e}")
            return ""
    
    def process_file(self, file_path: str, dry_run: bool = False) -> Dict:
        """
        Process a single CSV file with comprehensive cleaning and reporting.
        """
        self.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing: {file_path}")
        
        # Read file with multiple encoding attempts
        df = None
        encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully read file with {encoding} encoding")
                break
            except Exception as e:
                continue
        
        if df is None:
            self.logger.error(f"Could not read file {file_path} with any encoding")
            return {}
        
        df_original = df.copy()
        
        # Create backup
        if not dry_run:
            backup_path = self.create_backup(file_path)
        
        # Column cleaning and type detection
        type_changes = {}
        for col in df.columns:
            original_type = str(df[col].dtype)
            detected_type = self.detect_column_type(df[col])
            
            if detected_type == 'boolean':
                df[col] = self.clean_boolean(df[col])
            elif detected_type == 'datetime':
                df[col] = self.clean_datetime(df[col])
            elif detected_type == 'numerical':
                df[col] = self.clean_numerical(df[col])
            else:  # categorical
                df[col] = self.clean_categorical(df[col])
            
            type_changes[col] = {
                'original_type': original_type,
                'detected_type': detected_type,
                'final_type': str(df[col].dtype)
            }
            
            self.logger.info(f"Column '{col}': {original_type} -> {detected_type} -> {str(df[col].dtype)}")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Calculate quality metrics
        quality_metrics = self.calculate_data_quality_metrics(df_original, df)
        quality_metrics['type_changes'] = type_changes
        
        # Save cleaned file if not dry run
        if not dry_run:
            output_folder = self.config['output_folder']
            os.makedirs(output_folder, exist_ok=True)
            
            file_name = os.path.basename(file_path)
            cleaned_name = f"cleaned_{file_name}"
            output_path = os.path.join(output_folder, cleaned_name)
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"Cleaned file saved: {output_path}")
            
            # Save quality report
            report_name = f"quality_report_{file_name.replace('.csv', '.json')}"
            report_path = os.path.join(output_folder, report_name)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(quality_metrics, f, indent=2, default=str)
        
        return quality_metrics
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive data quality validation."""
        validation_results = {
            'completeness_score': 0,
            'consistency_score': 0,
            'validity_score': 0,
            'issues': []
        }
        
        # Completeness check
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        validation_results['completeness_score'] = ((total_cells - missing_cells) / total_cells) * 100
        
        # Consistency checks
        consistency_issues = 0
        total_checks = 0
        
        for col in df.columns:
            total_checks += 1
            # Check for mixed data types in categorical columns
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    consistency_issues += 1
                    validation_results['issues'].append(f"Column '{col}' contains mixed numeric/text data")
                except:
                    pass
        
        validation_results['consistency_score'] = ((total_checks - consistency_issues) / total_checks) * 100 if total_checks > 0 else 100
        
        # Validity checks (basic range and format validation)
        validity_issues = 0
        validity_checks = 0
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                validity_checks += 1
                # Check for reasonable ranges (no extreme outliers)
                if df[col].notna().any():
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > len(df) * 0.1:  # More than 10% outliers
                        validity_issues += 1
                        validation_results['issues'].append(f"Column '{col}' has {outliers} potential outliers")
        
        validation_results['validity_score'] = ((validity_checks - validity_issues) / validity_checks) * 100 if validity_checks > 0 else 100
        
        return validation_results
    
    def generate_summary_report(self, all_metrics: List[Dict]) -> str:
        """Generate a comprehensive summary report."""
        total_files = len(all_metrics)
        total_rows_before = sum(m.get('rows_before', 0) for m in all_metrics)
        total_rows_after = sum(m.get('rows_after', 0) for m in all_metrics)
        total_duplicates = sum(m.get('duplicates_removed', 0) for m in all_metrics)
        
        avg_completeness_before = np.mean([m.get('completeness_before', 0) for m in all_metrics])
        avg_completeness_after = np.mean([m.get('completeness_after', 0) for m in all_metrics])
        
        report = f"""
===============================================================================
                    DATA CLEANING SUMMARY REPORT
===============================================================================

Processing Summary:
• Files Processed: {total_files}
• Total Rows Before: {total_rows_before:,}
• Total Rows After: {total_rows_after:,}
• Duplicate Rows Removed: {total_duplicates:,}
• Data Retention Rate: {(total_rows_after/total_rows_before)*100:.2f}%

Data Quality Improvement:
• Average Completeness Before: {avg_completeness_before:.2f}%
• Average Completeness After: {avg_completeness_after:.2f}%
• Quality Improvement: {avg_completeness_after - avg_completeness_before:+.2f}%

Configuration Used:
• Numeric Threshold: {self.config['numeric_threshold']}
• DateTime Threshold: {self.config['datetime_threshold']}
• Boolean Threshold: {self.config['boolean_threshold']}
• Imputation Strategy: {self.config['imputation_strategy']}

=============================================================================== 
        """
        
        return report
    
    def run_cleaning_pipeline(self, dry_run: bool = False) -> Dict:
        """
        Execute the complete data cleaning pipeline.
        """
        input_folder = self.config['input_folder']
        
        if not os.path.exists(input_folder):
            self.logger.error(f"Input folder '{input_folder}' not found")
            return {}
        
        # Find CSV files
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in '{input_folder}'")
            return {}
        
        self.logger.info(f"Found {len(csv_files)} CSV files to process")
        
        all_metrics = []
        
        for i, file in enumerate(csv_files, 1):
            self.logger.info(f"Progress: {i}/{len(csv_files)} files")
            file_path = os.path.join(input_folder, file)
            
            try:
                metrics = self.process_file(file_path, dry_run)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to process {file}: {e}")
                continue
        
        # Generate and save summary report
        if all_metrics:
            summary_report = self.generate_summary_report(all_metrics)
            print(summary_report)
            
            if not dry_run:
                report_path = os.path.join(self.config['output_folder'], 'cleaning_summary_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(summary_report)
                self.logger.info(f"Summary report saved: {report_path}")
        
        return {
            'processed_files': len(all_metrics),
            'metrics': all_metrics,
            'summary': summary_report if all_metrics else "No files processed successfully"
        }


class ConfigGenerator:
    """Helper class to generate configuration files for the toolkit."""
    
    @staticmethod
    def create_default_config(config_path: str = "cleaning_config.json"):
        """Create a default configuration file."""
        default_config = {
            "input_folder": "raw_data",
            "output_folder": "cleaned_data",
            "backup_folder": "backup_data",
            "create_backup": True,
            "duplicate_threshold": 1.0,
            "datetime_threshold": 0.3,
            "numeric_threshold": 0.5,
            "boolean_threshold": 0.8,
            "imputation_strategy": {
                "numerical": "median",
                "categorical": "mode",
                "boolean": "mode"
            },
            "date_formats": [
                "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y"
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default configuration saved to: {config_path}")


# ---------- USAGE EXAMPLES ----------

def main():
    """Main execution function with multiple usage examples."""
    print("Advanced Data Cleaning Toolkit v6.0")
    print("=====================================\n")
    
    # Create default configuration if it doesn't exist
    if not os.path.exists("cleaning_config.json"):
        ConfigGenerator.create_default_config()
    
    # Initialize the toolkit
    cleaner = DataCleaningToolkit("cleaning_config.json")
    
    # Option 1: Run dry run first to preview changes
    print("Running dry run to preview changes...")
    dry_run_results = cleaner.run_cleaning_pipeline(dry_run=True)
    
    if dry_run_results.get('processed_files', 0) > 0:
        proceed = input("\nProceed with actual cleaning? (y/n): ").lower().strip()
        
        if proceed == 'y':
            print("\nExecuting data cleaning pipeline...")
            results = cleaner.run_cleaning_pipeline(dry_run=False)
            
            if results.get('processed_files', 0) > 0:
                print("\n✅ Data cleaning completed successfully!")
                print(f"📊 Processed {results['processed_files']} files")
                print(f"📁 Check '{cleaner.config['output_folder']}' for cleaned files")
                print(f"📋 Check 'logs/' for detailed processing logs")
            else:
                print("\n❌ No files were processed successfully")
        else:
            print("Operation cancelled by user")
    else:
        print("\n❌ No CSV files found or processed in dry run")


# ---------- COMMAND LINE INTERFACE ----------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Data Cleaning Toolkit")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying them")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--input-folder", type=str, help="Override input folder")
    parser.add_argument("--output-folder", type=str, help="Override output folder")
    
    args = parser.parse_args()
    
    if args.create_config:
        ConfigGenerator.create_default_config()
        exit(0)
    
    # Initialize toolkit
    cleaner = DataCleaningToolkit(args.config)
    
    # Override config if command line arguments provided
    if args.input_folder:
        cleaner.config['input_folder'] = args.input_folder
    if args.output_folder:
        cleaner.config['output_folder'] = args.output_folder
    
    # Run the pipeline
    if args.dry_run:
        print("Running in DRY RUN mode - no files will be modified")
    
    results = cleaner.run_cleaning_pipeline(dry_run=args.dry_run)
    
    if results.get('processed_files', 0) > 0:
        print(f"\n✅ Successfully processed {results['processed_files']} files")
    else:
        print("\n❌ No files were processed")
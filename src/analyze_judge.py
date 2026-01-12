import json
import os
import argparse
import logging
from collections import defaultdict
from datasets import load_dataset
import pandas as pd

from .constants import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def calculate_metrics(results):
    """Calculate metrics from grading results"""
    if not results:
        return {
            "total_samples": 0,
            "correct": 0,
            "incorrect": 0,
            "not_attempted": 0,
            "accuracy": 0.0,
            "accuracy_given_attempted": 0.0,
        }

    num_total = len(results)
    num_correct = sum(1 for r in results if r.get('is_correct', False))
    num_incorrect = sum(1 for r in results if r.get('is_incorrect', False))
    num_not_attempted = sum(1 for r in results if r.get('is_not_attempted', False))
    num_attempted = num_correct + num_incorrect

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    accuracy_given_attempted = num_correct / num_attempted if num_attempted > 0 else 0.0

    return {
        "total_samples": num_total,
        "correct": num_correct,
        "incorrect": num_incorrect,
        "not_attempted": num_not_attempted,
        "accuracy": accuracy,
        "accuracy_given_attempted": accuracy_given_attempted,
    }

def analyze_simpleqa_xl(graded_results, dataset):
    """Analyze simpleqa-xl results by subset_xl and language"""

    # Create mappings for dataset metadata
    metadata_map = {}
    for item in dataset:
        item_id = item.get('id', item.get('original_index'))
        metadata_map[item_id] = {
            'subset_xl': item.get('subset_xl', 'unknown'),
            'language': item.get('language', 'unknown')
        }

    # Group results by subset_xl and language
    subset_groups = defaultdict(list)
    language_groups = defaultdict(list)
    subset_language_groups = defaultdict(lambda: defaultdict(list))

    for result in graded_results:
        item_id = result['id']
        metadata = metadata_map.get(item_id, {'subset_xl': 'unknown', 'language': 'unknown'})

        subset_xl = metadata['subset_xl']
        language = metadata['language']

        subset_groups[subset_xl].append(result)
        language_groups[language].append(result)
        subset_language_groups[subset_xl][language].append(result)

    # Calculate metrics for each group
    subset_stats = {}
    for subset_xl, results in subset_groups.items():
        subset_stats[subset_xl] = calculate_metrics(results)

    language_stats = {}
    for language, results in language_groups.items():
        language_stats[language] = calculate_metrics(results)

    subset_language_stats = {}
    for subset_xl, lang_dict in subset_language_groups.items():
        subset_language_stats[subset_xl] = {}
        for language, results in lang_dict.items():
            subset_language_stats[subset_xl][language] = calculate_metrics(results)

    return {
        'by_subset': subset_stats,
        'by_language': language_stats,
        'by_subset_and_language': subset_language_stats
    }

def analyze_dataset(graded_file_path, dataset_type, seed):
    """Analyze a single graded results file"""

    # Load graded results
    with open(graded_file_path, 'r') as f:
        graded_results = json.load(f)

    # Calculate overall metrics
    overall_metrics = calculate_metrics(graded_results)

    analysis = {
        'dataset': dataset_type,
        'seed': seed,
        'overall': overall_metrics
    }

    # For simpleqa-xl, add detailed breakdowns
    if dataset_type == 'simpleqa-xl':
        # Load dataset for metadata
        dataset = load_dataset(EVAL_DATASETS_DICT[dataset_type], split='test')
        detailed_analysis = analyze_simpleqa_xl(graded_results, dataset)
        analysis.update(detailed_analysis)

    return analysis

def print_overall_summary(all_analyses):
    """Print overall summary table"""
    logging.info("\n" + "="*100)
    logging.info("OVERALL SUMMARY")
    logging.info("="*100)
    logging.info(f"{'Dataset':<20} {'Seed':<8} {'Total':<8} {'Correct':<8} {'Incorrect':<10} {'Not Att.':<10} {'Acc %':<10}")
    logging.info("-"*100)

    for analysis in all_analyses:
        overall = analysis['overall']
        logging.info(
            f"{analysis['dataset']:<20} "
            f"{analysis['seed']:<8} "
            f"{overall['total_samples']:<8} "
            f"{overall['correct']:<8} "
            f"{overall['incorrect']:<10} "
            f"{overall['not_attempted']:<10} "
            f"{overall['accuracy']*100:<10.2f} "
        )
    logging.info("="*100)

def print_subset_summary(analysis):
    """Print subset breakdown for simpleqa-xl"""
    if 'by_subset' not in analysis:
        return

    logging.info("\n" + "="*100)
    logging.info(f"SUBSET BREAKDOWN - {analysis['dataset'].upper()} (Seed {analysis['seed']})")
    logging.info("="*100)
    logging.info(f"{'Subset':<20} {'Total':<8} {'Correct':<8} {'Incorrect':<10} {'Not Att.':<10} {'Acc %':<10}")
    logging.info("-"*100)

    for subset_xl, metrics in sorted(analysis['by_subset'].items()):
        logging.info(
            f"{subset_xl:<20} "
            f"{metrics['total_samples']:<8} "
            f"{metrics['correct']:<8} "
            f"{metrics['incorrect']:<10} "
            f"{metrics['not_attempted']:<10} "
            f"{metrics['accuracy']*100:<10.2f} "
        )
    logging.info("="*100)

def print_language_summary(analysis):
    """Print language breakdown for simpleqa-xl"""
    if 'by_language' not in analysis:
        return

    logging.info("\n" + "="*100)
    logging.info(f"LANGUAGE BREAKDOWN - {analysis['dataset'].upper()} (Seed {analysis['seed']})")
    logging.info("="*100)
    logging.info(f"{'Language':<20} {'Total':<8} {'Correct':<8} {'Incorrect':<10} {'Not Att.':<10} {'Acc %':<10}")
    logging.info("-"*100)

    for language, metrics in sorted(analysis['by_language'].items()):
        logging.info(
            f"{language:<20} "
            f"{metrics['total_samples']:<8} "
            f"{metrics['correct']:<8} "
            f"{metrics['incorrect']:<10} "
            f"{metrics['not_attempted']:<10} "
            f"{metrics['accuracy']*100:<10.2f} "
        )
    logging.info("="*100)

def export_to_csv(all_analyses, output_dir):
    """Export statistics to CSV files"""

    # Overall statistics
    overall_data = []
    for analysis in all_analyses:
        row = {
            'dataset': analysis['dataset'],
            'seed': analysis['seed'],
            **analysis['overall']
        }
        overall_data.append(row)

    overall_df = pd.DataFrame(overall_data)
    overall_csv_path = os.path.join(output_dir, 'overall_statistics.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    logging.info(f"Overall statistics saved to: {overall_csv_path}")

    # Subset statistics (for simpleqa-xl)
    subset_data = []
    for analysis in all_analyses:
        if 'by_subset' in analysis:
            for subset_xl, metrics in analysis['by_subset'].items():
                row = {
                    'dataset': analysis['dataset'],
                    'seed': analysis['seed'],
                    'subset_xl': subset_xl,
                    **metrics
                }
                subset_data.append(row)

    if subset_data:
        subset_df = pd.DataFrame(subset_data)
        subset_csv_path = os.path.join(output_dir, 'subset_statistics.csv')
        subset_df.to_csv(subset_csv_path, index=False)
        logging.info(f"Subset statistics saved to: {subset_csv_path}")

    # Language statistics (for simpleqa-xl)
    language_data = []
    for analysis in all_analyses:
        if 'by_language' in analysis:
            for language, metrics in analysis['by_language'].items():
                row = {
                    'dataset': analysis['dataset'],
                    'seed': analysis['seed'],
                    'language': language,
                    **metrics
                }
                language_data.append(row)

    if language_data:
        language_df = pd.DataFrame(language_data)
        language_csv_path = os.path.join(output_dir, 'language_statistics.csv')
        language_df.to_csv(language_csv_path, index=False)
        logging.info(f"Language statistics saved to: {language_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze graded judge results')
    parser.add_argument('--graded_path', '-g', type=str, required=True,
                        help="Path to directory containing graded results JSON files")
    parser.add_argument('--dataset_names', '-d', type=str, required=True,
                        help="Comma-separated dataset names (e.g., simpleqa,simpleqa-xl)")
    parser.add_argument('--output_path', '-o', type=str, default='output/statistics',
                        help="Output directory for statistics files")
    parser.add_argument('--seeds_list', type=int, nargs='+', default=[0],
                        help="List of seeds to analyze")
    parser.add_argument('--show_detailed', action="store_true", dest="show_detailed",
                        help="Show detailed breakdowns (subset, language) for simpleqa-xl")
    parser.add_argument('--print_only', action="store_true", dest="print_only",
                        help="Show detailed printing break down; use this only after the first run.")
    parser.set_defaults(show_detailed=False, print_only=False)
    args = parser.parse_args()

    logging.info("==== Analysis Arguments ====")
    logging.info(args)
    logging.info("=== End of Arguments ====")

    # Parse dataset names
    dataset_names = args.dataset_names.strip()
    if dataset_names == "all":
        dataset_list = list(EVAL_DATASETS_DICT.keys())
    else:
        dataset_list = [name.strip() for name in dataset_names.split(",")]

    if args.print_only:
        json_output_path = os.path.join(args.output_path, 'all_statistics.json')
        with open(json_output_path, 'r') as f:
            all_analyses = json.load(f)
            
        # Extract entries
        simpleqa = next(x for x in all_analyses if x["dataset"] == "simpleqa")
        simpleqa_xl = next(x for x in all_analyses if x["dataset"] == "simpleqa-xl")

        base_acc = simpleqa["overall"]["accuracy"]

        # 1) Averages
        print(f"{simpleqa['overall']['accuracy'] * 100:.2f}")
        print(f"{simpleqa_xl['overall']['accuracy'] * 100:.2f}")
        print(f"{(simpleqa_xl['overall']['accuracy'] - base_acc) * 100:.2f}")

        # 2) By language (simpleqa-xl)
        for lang in sorted(simpleqa_xl.get("by_language", {})):
            acc = simpleqa_xl["by_language"][lang]["accuracy"]
            print(f"{acc * 100:.2f}")
            print(f"{(acc - base_acc) * 100:.2f}")

        # 3) By subset (simpleqa-xl)
        for subset in sorted(simpleqa_xl.get("by_subset", {})):
            acc = simpleqa_xl["by_subset"][subset]["accuracy"]
            print(f"{acc * 100:.2f}")
            print(f"{(acc - base_acc) * 100:.2f}")
    else:
        # Create output directory
        os.makedirs(args.output_path, exist_ok=True)

        # Analyze all datasets
        all_analyses = []

        for dataset_name in dataset_list:
            for seed in args.seeds_list:
                graded_file = os.path.join(args.graded_path, f"{dataset_name}_{seed}_graded.json")

                if not os.path.exists(graded_file):
                    logging.warning(f"Graded file not found: {graded_file}, skipping...")
                    continue

                logging.info(f"Analyzing: {dataset_name} (seed {seed})")
                analysis = analyze_dataset(graded_file, dataset_name, seed)
                all_analyses.append(analysis)

                # Show detailed breakdowns if requested
                if args.show_detailed and dataset_name == 'simpleqa-xl':
                    print_subset_summary(analysis)
                    print_language_summary(analysis)

        # Print overall summary
        if all_analyses:
            print_overall_summary(all_analyses)

            # Save to JSON
            json_output_path = os.path.join(args.output_path, 'all_statistics.json')
            with open(json_output_path, 'w') as f:
                json.dump(all_analyses, f, indent=2)
            logging.info(f"\nAll statistics saved to: {json_output_path}")

            # Export to CSV
            export_to_csv(all_analyses, args.output_path)
        else:
            logging.warning("No graded results found to analyze!")

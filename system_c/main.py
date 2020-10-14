import argparse
from scripts.combine_two_systems import CombineSystems

combine_systems = CombineSystems()

parser = argparse.ArgumentParser()

parser.add_argument("--a", type=str, default='files/system_a_output.csv', help="--a file_path")
parser.add_argument("--b", type=str, default='files/system-b_output.csv', help="--a file_path")
parser.add_argument("--dataset", type=str, default='site-a', help="--dataset dataset_type")
parser.add_argument("--output", type=str, default='files/output.csv', help="--output file_path")

args = parser.parse_args()
system_a_output = args.a
system_b_output = args.b
dataset = args.dataset
output_file = args.output

file_a_path = system_a_output
file_b_path = system_b_output
output_file_path = output_file

thresholds_a = {'environment': 0.1,
                'waiting time': 0.7,
                'staff attitude and professionalism': 0.4,
                'care quality': 0.2,
                'other': 0.6,
                }

thresholds_b = {'environment': 0.3,
                'waiting time': 0.1,
                'staff attitude and professionalism': 0.1,
                'care quality': 0.6,
                'other': 0.4
                }

if dataset == 'site-b':
    thresholds_a = {'environment': 0.6,
                    'waiting time': 0.5,
                    'staff attitude and professionalism': 0.5,
                    'care quality': 0.4,
                    'other': 0.7,
                    }

    thresholds_b = {'environment': 0.1,
                    'waiting time': 0.8,
                    'staff attitude and professionalism': 0.1,
                    'care quality': 0.1,
                    'other': 0.1
                    }

combine_systems.combine_by_dynamic_threshold(file_a_path, file_b_path, output_file_path, thresholds_a, thresholds_b)
combine_systems.extract_top_comments(file_a_path, '../top_comments_system_a.csv')
combine_systems.extract_top_comments(file_b_path, '../top_comments_system_b.csv')
print '\n\nThe final outputs saved successfully!\n'
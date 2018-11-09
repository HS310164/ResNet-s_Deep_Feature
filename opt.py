import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='input', type=str, help='Input videos dir')
    parser.add_argument('--output', '-o', default='outputs/', type=str, help='Output file dir')
    parser.add_argument('--only_hand', '-oh', action='store_true', help='If this option true, extract feature from region near hand')
    args = parser.parse_args()

    return args

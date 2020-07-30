import os
import argparse


def main():

    parser = argparse.ArgumentParser(description="Results convertion to LaTeX")

    parser.add_argument('--results', type=str, help='file with results', required=True)
    parser.add_argument('--output', type=str, help='output filename', required=True)

    args = parser.parse_args()

    p_results = args.results
    p_output = args.output


    with open(p_results, 'r') as f:
        lines = f.readlines()

    output_file = open(p_output, 'w')
    for line in lines:

        data = line.replace('\n', '').split(';')

        for index, v in enumerate(data):
            
            v = v.replace('_', '\_')

            try:
                v = "%.4f" % float(v)
            except ValueError:
                pass
                
            if index != len(data) - 1:
                output_file.write(v + ' & ')
            else:
                output_file.write(v + ' \\\\')

        output_file.write('\n\hline\n')

    output_file.close()

if __name__ == "__main__":
    main()
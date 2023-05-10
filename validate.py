import argparse
import pandas as pd


def main(args):
    gt = pd.read_json(args.gt, orient='records')
    predict = pd.read_json(args.predict, orient='records')

    results = gt.merge(predict, how='inner', on=['id', 'text', 'label'], suffixes=('_gt', '_predict'))
    assert len(results) == len(gt), 'Some texts from test dataset not found in predict json file, please check it manually'

    accuracy = (
            results['extracted_part_gt']
            .apply(lambda x: x['text'][0]) == results['extracted_part_predict']
            .apply(lambda x: x['text'][0])
    ).mean()
    print(f'Accuracy: {accuracy:0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', required=True, help='Path to file with predicted news labels.')
    parser.add_argument('--gt', required=True, help='Path to file with correct news labels.')
    args = parser.parse_args()

    main(args)
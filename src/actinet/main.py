import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='World')
    args = parser.parse_args()
    print(f'Hello {args.name}!')



if __name__ == '__main__':
    main()

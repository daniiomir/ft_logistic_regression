import src.tools as tools

if __name__ == '__main__':
    args = tools.parse_args_test()
    tools.read_dataset(args['dataset_path'])

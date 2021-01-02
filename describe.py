import src.tools as tools
import src.describe_tools as describe_tools

if __name__ == '__main__':
    args = tools.parse_args_describe()
    dataset = tools.read_dataset(args['dataset_path'])
    dsc = describe_tools.describe(dataset)
    print(dsc)

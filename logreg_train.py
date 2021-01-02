import src.tools as tools
from src.model import LogisticRegression

if __name__ == '__main__':
    args = tools.parse_args_train()
    dataset = tools.read_dataset(args['dataset_path'])
    # remove "Defense Against the Dark Arts" "Arithmancy" "Care of Magical Creatures"
    model = LogisticRegression()

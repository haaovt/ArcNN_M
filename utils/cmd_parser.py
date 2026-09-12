import argparse # for parsing command line arguments
import sys # for printing to stdout
from ruamel.yaml import YAML # for reading and writing yaml files

# this is to read terminal arguments

def get_agrs_parser(): # this function is to get the arguments from terminal, and also read the config file specified by the user, and print all the arguments and configurations in a formatted way for better readability
    parser = argparse.ArgumentParser(
        description="Parser Example",
        add_help=True
    )
    parser.add_argument('-c', '--config_file', help='Specify config file', metavar='FILE')
    subparser = parser.add_subparsers(dest='mode')
    parser_train = subparser.add_parser('train')
    parser_train.add_argument('--seed', type=int, default=0, metavar='N', help='Seed for data split and model initialization')
    parser_train.add_argument('--num_workers', type=int, default=4, metavar='N')
    parser_train.add_argument('--no_cuda', action='store_true', help='Do not use cuda (GPU).')

    args = parser.parse_args() # get all arguments in the parser
    arg_dict = vars(args) # convert the arguments to a dictionary for easier printing
    for key, value in arg_dict.items(): # print each argument and its value in a formatted way
        print(f'{key:25s} -> {value}')

    yaml = YAML() # for reading and writing yaml files, this is used to read the config file specified by the user, and print all the configurations in a formatted way for better readability
    yaml.indent(mapping = 2 , sequence=2, offset = 2) # this is to set the indentation of the yaml file for better readability, it adds 2 spaces before each line and also replaces the newline character with a newline character followed by 2 spaces
    yaml.default_flow_style = False # this is to set the flow style of the yaml file for better readability, it makes the yaml file more human-readable by using block style instead of flow style
    with open(args.config_file, 'r') as f: 
        cfg = yaml.load(f)
    yaml.dump(cfg, sys.stdout, transform=replace_indent) # this is to print the configurations in a formatted way for better readability, it adds 5 spaces before each line and also replaces the newline character with a newline character followed by 5 spaces

    return cfg, args

def replace_indent(stream): # this function is to replace the indentation of the yaml file for better readability, it adds 5 spaces before each line and also replaces the newline character with a newline character followed by 5 spaces
    stream = "     " + stream
    return stream.replace("\n", "\n     ")




import argparse
from distgen.drivers import run_distgen


def main():
    """
    Main function for running distgen as a single execution program
    """

    parser = argparse.ArgumentParser(description="Generate particle distributions")
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        default=None,
        help="Input configuration file",
        metavar="FILE",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        default=0,
        type=int,
        help="Verbosity level (default: 0)",
    )

    args = parser.parse_args()

    run_distgen(inputs=args.filename, verbose=args.verbose)


# ----------------------------------------------------------------------------
#   This allows the main function to be at the beginning of the file
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

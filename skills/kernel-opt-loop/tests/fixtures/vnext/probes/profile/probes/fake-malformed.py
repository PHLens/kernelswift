
import argparse, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    with open(args.result_json, "w", encoding="utf-8") as handle:
        handle.write("not json at all")
    return 0

if __name__ == "__main__":
    sys.exit(main())

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="survey.csv")
    args = parser.parse_args()

    args.file = "data/raw.csv"
    df = pd.read_csv(args.file)
    demos = []
    while True:
        # iv = input(
        #     "What independent variable do you want to include? (type 'done' to proceed to dependent variables)"
        # )
        demo_dict = {}
        iv = "gender"
        if iv == "done":
            break
        demo_dict["name"] = iv
        # cols = input(
        #     "What columns correspond to this variable? (comma-delimited, e.g., 'v102, v103, v104')"
        # ).split(",")
        cols = ["V201507x"]
        keys = []
        options = {}
        for col in cols:
            keys.append(col)
            # Find the unique values in the column and sort them
            unique_vals = sorted(df[col].unique())
            unique_val_ixs = range(len(unique_vals))
            ixd_unique_vals = {k: v for k, v in zip(unique_val_ixs, unique_vals)}
            # print each index and value
            for ix, val in ixd_unique_vals.items():
                print(ix, val)
            # ask user for a comma-delimited list of indexes for the values they want to include
            ixs = input(
                "Please give a comma-delimited list of values you want to include (e.g., 1, 2, 5 ) or a range (e.g., 1-5). The rest will be excluded"
            )
            if "-" in ixs:
                minimum, maximum = ixs.split("-")
                ixs = list(range(int(minimum), int(maximum) + 1))
            else:
                # convert the string to a list of ints
                ixs = [int(x) for x in ixs.split(",")]
            # Remove ixs from unique_val_ixs
            leftover_ixs = set(unique_val_ixs) - set(ixs)
            unacceptable_values = [unique_vals[ix] for ix in leftover_ixs]
            acceptable_values = [unique_vals[ix] for ix in ixs]
            options[col] = {}
            options[col]["invalid"] = unacceptable_values
            options[col]["valid"] = acceptable_values
        demo_dict["options"] = options
        demo_dict["keys"] = keys
        demos.append(demo_dict)

    while True:
        var = input(
            "What dependent variable do you want to include? (type 'done' to proceed)"
        )
        if var == "done":
            break
        # unique_vals = df[]
        unique_vals = df[var].unique()
        ixd_unique_vals = zip(range(len(unique_vals)), unique_vals)
        # print each index and value
        for ix, val in ixd_unique_vals:
            print(ix, val)
        # ask user for a comma-delimited list of indexes for the values they want to include
        ixs = input(
            "Please give a comma-delimited list of values you want to include (e.g., 1, 2, 5 ). The rest will be excluded"
        )
        # convert the string to a list of ints
        ixs = [int(x) for x in ixs.split(",")]
        return ixs

    print(df)


main()

import pandas as pd
import json


def create_dict_for_var(var, df):
    var_dict = {"name": var}
    keys = input(
        "What columns correspond to this variable? (comma-delimited, e.g., 'v102, v103, v104')"
    ).split(",")
    options = {}
    for key in keys:
        # Find the unique values in the key column and sort them
        unique_vals = sorted(df[key].unique())
        ixd_unique_vals = {i: val for i, val in enumerate(unique_vals)}
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
        leftover_ixs = set(range(len(unique_vals))) - set(ixs)
        invalid_options = [unique_vals[ix] for ix in leftover_ixs]
        valid_options = [unique_vals[ix] for ix in ixs]
        options[key] = {"invalid": invalid_options, "valid": valid_options}
    var_dict["options"] = options
    var_dict["keys"] = keys
    return var_dict


def main(args):
    df = pd.read_csv(args.file)
    # N.B., iv = independent variable
    ivs = []
    while True:
        iv = input(
            "Which independent variable do you want to include? (type 'done' to proceed to dependent variables)"
        )
        if iv == "done":
            break

        iv_dict = create_dict_for_var(iv, df)
        ivs.append(iv_dict)

    dvs = []
    while True:
        dv = input(
            "Which dependent variable do you want to include? (type 'done' to proceed)"
        )
        if dv == "done":
            break

        dv_dict = create_dict_for_var(dv, df)
        dvs.append(dv_dict)

    with open("data/ivs.json", "w") as f:
        json.dump(ivs, f)

    with open("data/dvs.json", "w") as f:
        json.dump(dvs, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    args = parser.parse_args()

    main(args)

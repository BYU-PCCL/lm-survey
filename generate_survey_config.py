import pandas as pd
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_dict_for_var(var, df):
    var_dict = {"name": var, "questions": {}}
    question_codes = input(
        "\nWhat columns correspond to this variable? (comma-delimited, e.g.,"
        " 'v102, v103, v104')\n:"
    ).split(",")
    for question_code in question_codes:
        # Find the unique values in the key column and sort them
        question_text = input(
            f"\nFor question {question_code}, what is the text for this"
            " question in the codebook?\n:"
        )

        unique_answer_codes = sorted(df[question_code].unique())
        ixd_unique_answer_codes = {
            ix: ans for ix, ans in enumerate(unique_answer_codes)
        }
        unique_answer_codes_str = "\n".join(
            f"{ix}, {code}" for ix, code in ixd_unique_answer_codes.items()
        )
        print(
            "\nFor that question, here are the unique values that answers"
            " assume, along with an index\nix,"
            f" val\n{unique_answer_codes_str}\n"
        )

        valid_ixs = input(
            "Please give a comma-delimited list of indices for values you want"
            " to include (e.g., 1, 2, 5 ) or a range (e.g., 1-5 inclusive). The"
            " rest will be excluded\n:"
        )
        if "-" in valid_ixs:
            minimum, maximum = valid_ixs.split("-")
            valid_ixs = list(range(int(minimum), int(maximum) + 1))
        else:
            # convert the string to a list of ints
            valid_ixs = [int(x) for x in valid_ixs.split(",")]
        # Remove ixs from unique_val_ixs
        invalid_ixs = set(range(len(unique_answer_codes))) - set(valid_ixs)

        invalid_options_dict = {
            ix: ans for ix, ans in ixd_unique_answer_codes.items() if ix in invalid_ixs
        }

        valid_options_dict = {}
        gen_format = input(
            "\nNow you will be asked whether you want to specify a general"
            " format for the data corresponding to this question to assume in"
            " your prompts and then whether there will be any exceptions to"
            ' that format. Do you want to specify a general format? (e.g. "I am'
            ' {X} years old" where {X} is a special token representing each'
            " specific answer)\nPress ENTER to skip\n:"
        )
        if gen_format == "":
            gen_format = "{X}"

        print(
            "Now you will be asked about the text in the codebook corresponding"
            " to each code, along with how to phrase each option to the"
            " language model\nPress ENTER to skip this option, and type 'skip'"
            " to start specifying exceptions to the general rule\n:"
        )
        skipping = False
        for ix in valid_ixs:
            valid_ans = ixd_unique_answer_codes[ix]
            if skipping:
                text = valid_ans
                rephrasing = valid_ans
            else:
                text = input(f"\nText corresponding to option '{valid_ans}': ")
                rephrasing = input(
                    f"Rephrasing for option '{valid_ans}' to appear in the"
                    f" {'{X}'} slot of your template (or to appear, if you"
                    " didn't write a template): "
                )
                if rephrasing == "skip":
                    skipping = True
                    rephrasing = valid_ans
                    text = valid_ans
                elif rephrasing == "":
                    rephrasing = valid_ans
                    text = valid_ans
            valid_options_dict[ix] = {
                "text": text,
                "phrased": gen_format.replace("{X}", rephrasing),
            }

        while True:
            # Ask user if there are any exceptions to the general format
            exception = input(
                "\nAre there any exceptions to the general format?\n(e.g., for"
                " the ix, val pairs \n\n0 Republican\n1 Democrat\n2"
                " Independent\n\nyou would type 2 to provide an exception for"
                " Independent)\nType 'done' to proceed to next"
                " variable or finish\n:"
            )

            if exception == "done":
                break
            else:
                exception = int(exception)
                text = input(
                    "What is the text corresponding to this code in the"
                    " codebook? Press ENTER to leave as is\n"
                )
                if text != "":

                    valid_options_dict[ixd_unique_answer_codes[exception]][
                        "text"
                    ] = text

                phrasing = input(
                    "How do you want to phrase this exception? Press ENTER to"
                    " leave as is\n"
                )
                if phrasing != "":
                    valid_options_dict[ixd_unique_answer_codes[exception]][
                        "phrasing"
                    ] = phrasing

        var_dict["questions"][question_code] = {
            "text": question_text,
            "valid_options": valid_options_dict,
            "invalid_options": invalid_options_dict,
        }

    return var_dict


def main(data_filename):
    df = pd.read_csv(data_filename)
    var_list = []
    while True:
        iv = input(
            "Which variables do you want to include (e.g., 'gender')? (type"
            " 'done' to finish and write a json objects for the variables"
            " you've specified)\n:"
        )
        if iv == "done":
            break

        var = create_dict_for_var(iv, df)
        var_list.append(var)

    with open("data/vars.json", "w") as f:
        json.dump(var_list, f, cls=NpEncoder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_filename", type=str, default="data/roper/data.csv"
    )
    args = parser.parse_args()

    main(data_filename=args.data_filename)

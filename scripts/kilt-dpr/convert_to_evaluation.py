import argparse
import json
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kilt_queries_file", type=str)
    parser.add_argument("--provenance_file", type=str)
    parser.add_argument("--output_evaluation_file", type=str)
    args = parser.parse_args()

    raw_data = []
    with open(args.kilt_queries_file, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))

    with open(args.provenance_file, "r") as f:
        provenance = json.load(f)

    # consider only valid data - filter out invalid
    validated_data = {}
    query_data = []
    for element in raw_data:
        #if utils.validate_datapoint(element, logger=None):
        if element["id"] in validated_data:
            raise ValueError("ids are not unique in input data!")
        validated_data[element["id"]] = element
        query_data.append(
            {"query": element["input"], "id": element["id"]}
        )

    if len(provenance) != len(query_data):
        print("WARNING: provenance and query data are not of the same length!")

    # write prediction files
    if provenance:
        print("writing prediction file to {}".format(args.output_evaluation_file))

        predictions = []
        for query_id in provenance.keys():
            element = validated_data[query_id]
            new_output = [{"provenance": provenance[query_id]}]
            # append the answers
            if "output" in element:
                for o in element["output"]:
                    if "answer" in o:
                        new_output.append({"answer": o["answer"]})
            element["output"] = new_output
            predictions.append(element)

    with open(args.output_evaluation_file, "w") as outfile:
        for p in predictions:
            json.dump(p, outfile)
            outfile.write("\n")
import os

import numpy as np
import pandas as pd
import yaml

import wandb
from .....utils.charts import get_LLM_order, get_pretty_name_llm

from .utils import generate_colors


def flatten_yaml(yaml_object, parent_key="", sep="."):
    """
    Flatten a nested YAML file with a recursive function.
    The keys are concatenated to represent the hierarchy.
    """
    items = []
    for k, v in yaml_object.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_yaml(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(
                        flatten_yaml(item, f"{new_key}{sep}{i}", sep=sep).items()
                    )
                else:
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def columns_non_relevant(df):
    columns_with_variability = []
    for column in df.columns:
        if all(isinstance(x, list) for x in df[column]):
            # Convert each list to a tuple for comparison since tuples are hashable
            unique_tuples = set(tuple(x) for x in df[column])
            if len(unique_tuples) == 1:
                columns_with_variability.append(column)
        else:
            if df[column].nunique() == 1:
                columns_with_variability.append(column)
    return columns_with_variability


def get_summary_runs(subset_name, WANDB=False):
    # Collect all runs data

    acc = []
    if subset_name is None:
        subset_name = "venture_adversary_universalization_low_earn"

    if "final" in subset_name:
        WANDB = False

    if WANDB:
        api = wandb.Api(timeout=30)
        runs = api.runs("EMS", filters={"tags": {"$nin": ["skip"]}})
        for r in runs:
            if (
                r.state == "crashed" or r.state == "failed"
            ) and "include" not in r.tags:
                continue

            if subset_name not in r.group:
                # print("skip1")
                continue

            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../results",
                r.group,
                r.name,
                "log_env.json",
            )
            if not os.path.exists(log_path):
                log_dir = os.path.dirname(log_path)
                os.makedirs(log_dir, exist_ok=True)
                run_files = r.files()
                for run_file in run_files:
                    print(run_file.name)
                    if run_file.name == "log_env.json":
                        print(f"Downloading: {r.name}")
                        run_file.download(root=log_dir)
                # continue

            run_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../results",
                r.group,
                r.name,
                ".hydra/config.yaml",
            )
            flat_data = {}
            if os.path.exists(run_path):
                # Read the YAML file
                with open(run_path, "r") as file:
                    yaml_data = yaml.safe_load(file)

                # Flatten the YAML data
                flat_data = flatten_yaml(yaml_data)
                flat_data = {k: [v] for k, v in flat_data.items()}
            else:
                # trie to read the config from wandb
                flat_data = flatten_yaml(r.config)
                flat_data = {k: [v] for k, v in flat_data.items()}
                to_remove = [
                    "tags.1",
                    "tags.2",
                    "tags.3",
                    "tags.4",
                    "tags.5",
                    "experiment.name",
                    "experiment.agent.converse.inject_resource_observation",
                ]
                for k in to_remove:
                    if k in flat_data:
                        del flat_data[k]
            # print(f"Flat data: {flat_data}")
            acc.append(
                pd.DataFrame(
                    {
                        "name": [r.name],
                        "group": [flat_data["llm.path"][0]],
                        "run_group": [r.group],
                        **flat_data,
                    },
                    index=[len(acc)],
                )
            )
    else:
        base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "results")
)
        for group in os.listdir(base_path):
            group_path = os.path.join(base_path, group)
            if not os.path.isdir(group_path) or subset_name not in group:
                continue
            for run in os.listdir(group_path):

                run_path = os.path.join(group_path, run)
                if not os.path.isdir(run_path):
                    continue

                # from here should be the same as the wandb part
                log_path = os.path.join(run_path, "log_env.json")
                if not os.path.exists(log_path):
                    continue
                run_path = os.path.join(run_path, ".hydra/config.yaml")
                flat_data = {}
                if os.path.exists(run_path):
                    # Read the YAML file
                    with open(run_path, "r") as file:
                        yaml_data = yaml.safe_load(file)

                    # Flatten the YAML data
                    flat_data = flatten_yaml(yaml_data)
                    flat_data = {k: [v] for k, v in flat_data.items()}
                else:
                    continue  # no wandb access!

                acc.append(
                    pd.DataFrame(
                        {
                            "name": [run],
                            "group": [flat_data["llm.path"][0]],
                            "run_group": [group],
                            **flat_data,
                        },
                        index=[len(acc)],
                    )
                )

    summary_df = pd.concat(acc)
    non_relevant = columns_non_relevant(summary_df)
    if len(summary_df) <= 1:
        non_relevant = []
    # remove "group" from non_relevant
    non_relevant = [
        c
        for c in non_relevant
        if c != "group"
        and c != "run_group"
        and c != "group_name"
        # and c != "experiment.env.name"
        and c != "llm.path"
        and c != "llm.is_api"
    ]
    cols = [c for c in summary_df.columns if c.startswith("experiment.personas.")]
    cols.append("experiment.agent.name")
    cols.append("llm.top_p")
    cols.append("seed")
    cols.append("experiment.env.num_agents")
    # cols.append("experiment.env.class_name")
    cols.extend(non_relevant)
    summary_df = summary_df.drop(columns=cols, errors="ignore")
    summary_df = summary_df.rename(
        columns={
            # "experiment.env.name": "env",
            "experiment.agent.act.universalization_prompt": "universalization_prompt",
            "experiment.agent.act.harvest_strategy": "harvest_strategy",
            "experiment.agent.act.consider_identity_persona": (
                "consider_identity_persona"
            ),
            "llm.temperature": "temperature",
        }
    )
    summary_group_df = (
        summary_df.drop_duplicates(subset=["group"])
        .set_index("group", drop=True)
    )
    summary_group_df["id"] = summary_group_df.index

    # filter, remove models that have GPTQ in the llm.path
    # summary_group_df = summary_group_df[
    #     ~summary_group_df["llm.path"].str.contains("GPTQ")
    # ]

    order = get_LLM_order()
    # sort row based on llm.path col, according to the above array of names
    summary_group_df["order_col"] = summary_group_df["llm.path"].apply(
        get_pretty_name_llm
    )
    summary_group_df["order_col"] = pd.Categorical(
        summary_group_df["order_col"], categories=order, ordered=True
    )
    # Now sort the DataFrame by 'llm.path'
    summary_group_df = summary_group_df.sort_values("order_col")
    # drop the order_col
    summary_group_df = summary_group_df.drop(columns=["order_col"])
    summary_df.to_csv("summary.csv")
    summary_group_df.to_csv("summary_group.csv")
    return summary_df, summary_group_df

def load_runs_data(summary_df, summary_group_df):
    # print(f"Load Summary DF: {summary_df}")
    base_path = os.path.dirname(os.path.abspath(__file__))

    run_data = {group: {} for group in summary_df["group"].unique()}
    # print(f"Run data1: {run_data}")
    for id, row in summary_df.iterrows():
        run_path = os.path.join(
            base_path, "../../../results", row["run_group"], row["name"], "log_env.json"
        )
        run_data[row["group"]][row["name"]] = pd.read_json(run_path, orient="records")

    # print(f"Run data: {run_data}")

    print(f"Summary group df: {summary_group_df}")
    return {
        "summary_group_df": summary_group_df,  # .to_dict(orient="records"),
        "summary_df": summary_df,
        "run_data": run_data,
    }


def get_data(subset_name):
    summary_df, summary_group_df = get_summary_runs(subset_name, WANDB=False)
    colours = generate_colors(len(summary_group_df))
    summary_group_df["colour_0"] = [c[0] for c in colours]
    summary_group_df["colour_1"] = [c[1] for c in colours]
    data = load_runs_data(summary_df, summary_group_df)
    # print(f"Data {data}")
    return data

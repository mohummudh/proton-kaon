import pandas as pd
import uproot

REQUIRED_BRANCHES = {"run", "subrun", "event"}


def _select_tree(root_file, tree_name=None):
    if tree_name is not None:
        return root_file[tree_name]

    tree_keys = root_file.keys(
        recursive=True,
        cycle=False,
        filter_classname="TTree",
    )

    if not tree_keys:
        raise ValueError("No TTree found in file")

    for key in tree_keys:
        tree = root_file[key]
        if REQUIRED_BRANCHES.issubset(tree.keys()):
            return tree

    if len(tree_keys) == 1:
        return root_file[tree_keys[0]]

    raise ValueError(
        f"Multiple TTrees found {tree_keys}, and none matched {sorted(REQUIRED_BRANCHES)}. "
        "Pass tree_name explicitly."
    )


def open_root(file_path, tree_name=None):
    root_file = uproot.open(file_path)
    tree = _select_tree(root_file, tree_name=tree_name)

    n = tree.num_entries
    run = tree["run"].array(library="np")
    subrun = tree["subrun"].array(library="np")
    event = tree["event"].array(library="np")

    return pd.DataFrame({
        "file_path": [file_path] * n,
        "event_index": list(range(n)),
        "run": run,
        "subrun": subrun,
        "event": event,
    })

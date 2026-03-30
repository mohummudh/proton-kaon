import uproot
import pandas as pd

def open_root(file_path, ):

    file = file_path
    tree = uproot.open(file)["ana/raw"]

    n = tree.num_entries   # should be 16036 in your case

    # load run/subrun/event (optional but recommended)
    run    = tree["run"].array(library="np")
    subrun = tree["subrun"].array(library="np")
    event  = tree["event"].array(library="np")

    events_df = pd.DataFrame({
        "file_path": [file] * n,
        "event_index": list(range(n)),
        "run": run,
        "subrun": subrun,
        "event": event,
    })

    return events_df
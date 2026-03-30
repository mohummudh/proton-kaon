import uproot
import numpy as np
import pandas as pd

from tqdm import tqdm
from src.event import Event
from src.open_root import _select_tree

def extract_clusters(events_df, particle_type, threshold=15, max_events=None, tree_name=None):
    """Extract all clusters from events and create a pandas dataframe"""
    
    cluster_data = []

    if max_events:
        events_df = events_df.head(max_events)

    file_path = events_df["file_path"].iat[0]
    root_file = uproot.open(file_path)
    tree = _select_tree(root_file, tree_name=tree_name)
    
    for i, row in tqdm(events_df.iterrows(), total=len(events_df)):

        

        try:
            # Create event
            event = Event(tree=tree, filepath=row.file_path, index=row.event_index, plot=False)
            
            # Get connected regions for collection plane
            clabeled, cregions = event.connectedregions(event.collection, threshold=threshold)
            ilabeled, iregions = event.connectedregions(event.induction, threshold=(threshold // 2))
            
            # Process collection plane clusters
            if cregions is not None:
                for j, region in enumerate(cregions): # index, element 

                    column_maxes = region.image_intensity.max(axis=1)
                    
                    cluster_info = {
                        'event_idx': i,
                        'run': row.run,
                        'subrun': row.subrun,
                        'event': row.event,
                        'file_path': row.file_path,
                        'event_index': row.event_index,
                        'particle_type': particle_type,
                        'plane': 'collection',
                        'cluster_idx': j,
                        'bbox_min_row': region.bbox[0],
                        'bbox_min_col': region.bbox[1],
                        'bbox_max_row': region.bbox[2],
                        'bbox_max_col': region.bbox[3],
                        'width': region.bbox[3] - region.bbox[1],
                        'height': region.bbox[2] - region.bbox[0],
                        'image_intensity': region.image_intensity,  # Original image
                        'column_maxes': column_maxes               # Column maxes array
                    }
                    cluster_data.append(cluster_info)
            
            # Process induction plane clusters
            if iregions is not None:
                for j, region in enumerate(iregions):

                    column_maxes = region.image_intensity.max(axis=1)
                    
                    cluster_info = {
                        'event_idx': i,
                        'run': row.run,
                        'subrun': row.subrun,
                        'event': row.event,
                        'file_path': row.file_path,
                        'event_index': row.event_index,
                        'particle_type': particle_type,
                        'plane': 'induction',
                        'cluster_idx': j,
                        'bbox_min_row': region.bbox[0],
                        'bbox_min_col': region.bbox[1],
                        'bbox_max_row': region.bbox[2],
                        'bbox_max_col': region.bbox[3],
                        'width': region.bbox[3] - region.bbox[1],
                        'height': region.bbox[2] - region.bbox[0],
                        'image_intensity': region.image_intensity,  # Original image
                        'column_maxes': column_maxes               # Column maxes array
                    }
                    cluster_data.append(cluster_info)
                    
        except Exception as e:
            print(f"Error processing event {i}: {e}")
            continue
    
    return pd.DataFrame(cluster_data)
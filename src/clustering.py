import numpy as np
import pandas as pd

from tqdm import tqdm
from event import Event

def extract_clusters(events_df, particle_type, threshold=15, max_events=None):
    """Extract all clusters from events and create a pandas dataframe"""
    
    cluster_data = []

    if max_events:
        events_df = events_df.head(max_events)
    
    for i, row in tqdm(events_df.iterrows(), total=len(events_df)): # iterrows gives index, Series (row)
        try:
            # Create event
            event = Event(row.file_path, index=row.event_index, plot=False)
            
            # Get connected regions for collection plane
            clabeled, cregions = event.connectedregions(event.collection, threshold=threshold)
            ilabeled, iregions = event.connectedregions(event.induction, threshold=threshold)
            
            # Process collection plane clusters
            if cregions is not None:
                for j, region in enumerate(cregions): # index, element 

                    matrix = region.image_intensity 
                    matrix_transformed = matrix.T[::-1] # image of cluster 
                    column_maxes = np.max(matrix_transformed, axis=0) # 1D matrix, max ADC for each wire in cluster - gives a 1D view of energy deposition
                    
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
                    # Get the matrix and column maxes
                    matrix = region.image_intensity
                    matrix_transformed = matrix.T[::-1]
                    column_maxes = np.max(matrix_transformed, axis=0)
                    
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

def read_value_labels(f, features):
    """
    Reads a set of features and a label from a file one row at a time.
    """
    for line in f: # Implicitly splits rows on \n
        parts = line.strip().split("\t") # Splits columns on \t
        values = parts[:-1] # All but the last column are feature values.
        label = parts[-1] # Last column is a label

        feature_values = []
        for feature, value in zip(features, values): 
            # Each feature knows its type and will perform the right conversion
            
            if feature.returns == bool:
                # Booleans are weird.  bool("False") == True, so you need to string match "True"
                feature_values.append(value == "True")
            else:
                feature_values.append(feature.returns(value))

        row = feature_values[:]
        row.append(label == "True")
                
        yield row
  
from editquality.feature_lists import enwiki

f = open("enwiki_data/enwiki.features_damaging.20k_2015.tsv")
rows = list(read_value_labels(f, enwiki.damaging))
import numpy
data_mat = numpy.array(rows).astype(float)
import sigclust.sigclust
result = sigclust.sigclust.sigclust(data_mat)

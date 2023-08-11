import numpy as np

class Constraint:
    def enforce(self, weights):
        # weights is np.array
        return weights
    
    
class LongOnly(Constraint):
    def enforce(self, weights):
        # only positive weights
        weights[weights < 0] = 0
        
        # weights sum to 1
        return weights / weights.sum(axis = 1, keepdims = True)
    
    
    
    
# leverage, market neutral, ?


# old code:
    
    
def enforce_constraints(weights, constraint_val = None, constraint_type = "Eq"):
    weights = enforce_standard_constraint(weights)
    
    if constraint_val is not None:
        if constraint_type == "Eq":
            weights = enforce_equality_constraint(weights, constraint_val)
        elif constraint_type == "Le":
            weights = enforce_less_than_constraint(weights, constraint_val)
        else:
            weights = enforce_greater_than_constraint(weights, constraint_val)
    
    return weights
    

def enforce_standard_constraint(weights):
    weights[weights < 0] = 0
    return weights / weights.sum(axis = 1, keepdims = True)

def enforce_equality_constraint(weights, value):
    # assumes constrained variable is in the first position
    
    weights[:, 0] = value
    weights[:, 1:] /= weights[:, 1:].sum(axis = 1, keepdims = True)
    weights[:, 1:] *= 1 - value
    
    return weights

def enforce_greater_than_constraint(weights, value):
    # assumes constrained variable is in the first position
    
    weights[:, 0] = np.maximum(weights[:, 0], value)
    weights[:, 1:] /= weights[:, 1:].sum(axis = 1, keepdims = True)
    weights[:, 1:] *= 1 - weights[:  0]
    
    return weights

def enforce_less_than_constraint(weights, value):
    # assumes constrained variable is in the first index

    weights[:, 0] = np.minimum(weights[:, 0], value)
    weights[:, 1:] /= weights[:, 1:].sum(axis=1, keepdims=True)
    weights[:, 1:] *= 1 - weights[:, 0]
    
    return weights
    
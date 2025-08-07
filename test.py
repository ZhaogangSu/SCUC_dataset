import numpy as np
from coptpy import *

def create_markshare2_base():
    """Create the base markshare2 model"""
    
    # Create COPT environment and model
    env = Envr()
    model = env.createModel("markshare2")
    
    # Coefficient matrix A (7x60)
    A = np.array([
        [74,49,12,93,56,16,39,77,56,73,1,3,68,61,8,55,18,21,57,98,58,57,46,72,6,16,76,21,78,18,11,58,59,25,32,14,16,3,60,12,7,42,98,34,33,16,97,63,66,28,57,19,74,44,45,49,76,74,9,44],
        [20,7,68,69,95,64,76,12,45,43,83,15,90,10,96,98,53,1,2,58,24,90,29,57,19,73,89,31,12,34,67,48,11,22,36,78,75,52,95,57,62,94,10,42,89,11,77,85,30,82,20,52,78,6,57,65,79,83,16,67],
        [85,47,67,59,84,59,19,8,50,66,5,51,51,64,64,53,61,45,3,76,17,54,13,89,68,57,4,24,96,81,36,54,3,82,33,88,1,29,4,48,51,14,86,64,73,78,45,65,30,52,6,78,9,19,87,73,10,87,33,1],
        [13,71,78,84,56,66,8,68,48,28,33,34,8,99,80,74,2,10,96,41,98,74,39,91,85,95,96,1,80,90,97,36,7,69,9,9,93,94,44,36,71,37,72,38,74,89,37,24,88,77,61,80,2,60,87,80,74,42,2,37],
        [35,61,66,78,46,89,61,25,55,16,81,35,96,23,83,39,14,53,23,23,93,38,15,20,19,28,79,51,24,6,3,47,61,60,71,63,26,66,71,63,56,32,39,31,64,89,62,68,59,71,48,76,96,74,61,21,46,18,23,24],
        [86,8,44,96,64,65,68,53,19,33,28,42,72,39,5,77,37,89,7,78,10,78,10,96,55,1,64,61,63,90,22,78,92,25,24,65,6,68,66,66,1,67,78,21,47,17,89,77,88,54,10,87,88,80,76,9,83,95,86,24],
        [41,64,82,24,48,41,29,93,64,39,92,86,64,45,87,34,39,88,99,63,85,48,83,88,85,5,14,31,12,93,55,1,2,22,93,49,35,25,39,1,77,43,7,42,36,63,5,8,43,18,60,47,47,46,45,38,9,37,8,82]
    ])
    
    # RHS values
    b = np.array([1324, 1554, 1429, 1686, 1482, 1613, 1424])
    
    # Create variables
    x = []  # Binary variables
    for j in range(60):
        x.append(model.addVar(vtype=COPT.BINARY, name=f"x{j+15}"))
    
    y = []  # Deficit variables
    for i in range(7):
        y.append(model.addVar(lb=0, vtype=COPT.CONTINUOUS, name=f"y{i+1}"))
    
    # Add base constraints: A*x + y = b
    for i in range(7):
        expr = LinExpr()
        for j in range(60):
            expr += A[i,j] * x[j]
        expr += y[i]
        model.addConstr(expr == b[i], name=f"c{i+1}")
    
    # Set objective: minimize sum of deficits
    obj = LinExpr()
    for i in range(7):
        obj += y[i]
    model.setObjective(obj, COPT.MINIMIZE)
    
    return model, x, y, A, b

def create_markshare2_binary_decomposition():
    """
    Binary decomposition reformulation
    Since y must be integer (all coeffs and RHS are integer), we decompose y into binary representation
    """
    
    # Create COPT environment and model
    env = Envr()
    model = env.createModel("markshare2_binary")
    
    # Same coefficient matrix and RHS
    A = np.array([
        [74,49,12,93,56,16,39,77,56,73,1,3,68,61,8,55,18,21,57,98,58,57,46,72,6,16,76,21,78,18,11,58,59,25,32,14,16,3,60,12,7,42,98,34,33,16,97,63,66,28,57,19,74,44,45,49,76,74,9,44],
        [20,7,68,69,95,64,76,12,45,43,83,15,90,10,96,98,53,1,2,58,24,90,29,57,19,73,89,31,12,34,67,48,11,22,36,78,75,52,95,57,62,94,10,42,89,11,77,85,30,82,20,52,78,6,57,65,79,83,16,67],
        [85,47,67,59,84,59,19,8,50,66,5,51,51,64,64,53,61,45,3,76,17,54,13,89,68,57,4,24,96,81,36,54,3,82,33,88,1,29,4,48,51,14,86,64,73,78,45,65,30,52,6,78,9,19,87,73,10,87,33,1],
        [13,71,78,84,56,66,8,68,48,28,33,34,8,99,80,74,2,10,96,41,98,74,39,91,85,95,96,1,80,90,97,36,7,69,9,9,93,94,44,36,71,37,72,38,74,89,37,24,88,77,61,80,2,60,87,80,74,42,2,37],
        [35,61,66,78,46,89,61,25,55,16,81,35,96,23,83,39,14,53,23,23,93,38,15,20,19,28,79,51,24,6,3,47,61,60,71,63,26,66,71,63,56,32,39,31,64,89,62,68,59,71,48,76,96,74,61,21,46,18,23,24],
        [86,8,44,96,64,65,68,53,19,33,28,42,72,39,5,77,37,89,7,78,10,78,10,96,55,1,64,61,63,90,22,78,92,25,24,65,6,68,66,66,1,67,78,21,47,17,89,77,88,54,10,87,88,80,76,9,83,95,86,24],
        [41,64,82,24,48,41,29,93,64,39,92,86,64,45,87,34,39,88,99,63,85,48,83,88,85,5,14,31,12,93,55,1,2,22,93,49,35,25,39,1,77,43,7,42,36,63,5,8,43,18,60,47,47,46,45,38,9,37,8,82]
    ])
    
    b = np.array([1324, 1554, 1429, 1686, 1482, 1613, 1424])
    
    # Binary selection variables
    x = []
    for j in range(60):
        x.append(model.addVar(vtype=COPT.BINARY, name=f"x{j+1}"))
    
    # Binary decomposition of deficit: y[i] = sum(2^k * y_bit[i,k])
    # 8 bits can represent 0-255, enough for any reasonable deficit
    y_bit = {}
    for i in range(7):
        for k in range(8):
            y_bit[i,k] = model.addVar(vtype=COPT.BINARY, name=f"y_bit_{i+1}_{k}")
    
    # Carry variables for binary arithmetic
    # carry[i,k] is the carry from bit k to bit k+1 for constraint i
    carry = {}
    for i in range(7):
        for k in range(7):  # carry from bit 0-6 to bit 1-7
            carry[i,k] = model.addVar(lb=0, vtype=COPT.INTEGER, name=f"carry_{i+1}_{k}")
    
    # Binary decomposition constraints
    for i in range(7):
        for k in range(8):
            # Calculate the sum at bit position k
            bit_sum_expr = LinExpr()
            
            # Add contribution from A*x at bit position k
            for j in range(60):
                # Get bit k of A[i,j]
                a_bit_k = (A[i,j] >> k) & 1
                if a_bit_k == 1:
                    bit_sum_expr += x[j]
            
            # Add y_bit contribution
            bit_sum_expr += y_bit[i,k]
            
            # Add carry from previous bit (if exists)
            if k > 0:
                bit_sum_expr += carry[i,k-1]
            
            # Get bit k of b[i]
            b_bit_k = (b[i] >> k) & 1
            
            if k < 7:
                # Standard bit equation with carry
                # bit_sum = b_bit_k + 2 * carry[i,k]
                model.addConstr(bit_sum_expr == b_bit_k + 2 * carry[i,k], 
                              name=f"bit_constraint_{i+1}_{k}")
            else:
                # Last bit: no carry out
                # bit_sum = b_bit_k
                model.addConstr(bit_sum_expr == b_bit_k, 
                              name=f"bit_constraint_{i+1}_{k}")
    
    # Objective: minimize sum of deficits
    # deficit[i] = sum(2^k * y_bit[i,k])
    obj = LinExpr()
    for i in range(7):
        for k in range(8):
            obj += (2**k) * y_bit[i,k]
    
    model.setObjective(obj, COPT.MINIMIZE)
    
    return model, x, y_bit, carry



